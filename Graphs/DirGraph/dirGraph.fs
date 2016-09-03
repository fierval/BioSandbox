namespace Graphs

open System.Collections.Generic
open System.Linq
open System.IO
open System
open DrawGraph
open System.Linq
open Alea.CUDA

#nowarn "25"

open GpuGoodies

/// <summary>
/// Instantiate a directed graph. Need number of vertices
/// Format of the file:
/// v -> a, b, c, d - v - unique vertex name for each line of the file. a, b, c, d - names of vertices it connects to.
/// </summary>
[<StructuredFormatDisplay("{AsEnumerable}")>]
type DirectedGraph<'a when 'a:comparison> (rowIndex : int seq, colIndex : int seq, verticesNameToOrdinal : IDictionary<'a, int>) as this =
    let rowIndex  = rowIndex.ToArray()
    let colIndex = colIndex.ToArray()
    let nEdges = colIndex.Length
    let verticesNameToOrdinal = verticesNameToOrdinal
    let nVertices = verticesNameToOrdinal.Count

    let ordinalToNames () =
        let res : 'a [] = Array.zeroCreate verticesNameToOrdinal.Count
        verticesNameToOrdinal
        |> Seq.iter (fun kvp -> res.[kvp.Value] <- kvp.Key)

        res

    let verticesOrdinalToNames = ordinalToNames()

    let nameFromOrdinal = fun ordinal -> verticesOrdinalToNames.[ordinal]
    let ordinalFromName = fun name -> verticesNameToOrdinal.[name]

    // vertices connected to the ordinal vertex
    let getVertexConnections ordinal =
        let start = rowIndex.[ordinal]
        let end' = rowIndex.[ordinal + 1] - 1
        colIndex.[start..end']

    // vertices and edge #'s of vertices
    // connected to the "ordinal" vertex
    let getVerticesAndEdges ordinal =
        let start = rowIndex.[ordinal]
        let end' = rowIndex.[ordinal + 1] - 1
        colIndex.[start..end'], [|start..end'|]

    let asOrdinalsEnumerable () =
        Seq.init nVertices (fun i -> i, getVertexConnections i)

    let reverse =
        lazy (
            let allExistingRows = [0..rowIndex.Length - 1]

            let subSeq =
                if hasCuda.Force() && rowIndex.Length >= gpuThresh then //use CUDA to reverse
                    let start, end' =
                        let dStart, dEnd = getEdgesGpu rowIndex colIndex
                        sortStartEnd dStart dEnd

                    Seq.zip end' start
                else
                    asOrdinalsEnumerable ()
                    |> Seq.map (fun (i, verts) -> verts |> Seq.map (fun v -> (v, i)))
                    |> Seq.collect id

            let grSeq =
                subSeq
                |> Seq.groupBy fst
                |> Seq.map (fun (key, sq) -> key, sq |> Seq.map snd |> Seq.toArray)

            let allRows : seq<int * int []> =
                allExistingRows.Except (grSeq |> Seq.map fst) |> Seq.map (fun e -> e, [||])
                |> fun col -> col.Union grSeq
                |> Seq.sortBy fst


            let revRowIndex = allRows |> Seq.scan (fun st (key, v) -> st + v.Length) 0 |> Seq.take rowIndex.Length
            let revColIndex = allRows |> Seq.collect snd

            DirectedGraph(revRowIndex, revColIndex, verticesNameToOrdinal)
        )

    let hasEulerPath =
        lazy (
            // start: out = in + 1
            let rows = Array.zip rowIndex (reverse.Force().RowIndex)
            let pref = Array.takeWhile (fun (gr, rev) -> gr = rev) rows
            let middle = Array.takeWhile (fun (gr, rev) -> abs(gr - rev) = 1) rows.[pref.Length..]
            let remains = Array.takeWhile (fun (gr, rev) -> gr = rev) rows.[pref.Length + middle.Length..]
            pref.Length <> rows.Length && remains.Length + middle.Length + pref.Length = rows.Length
        )

    let partitionLinear =
        lazy (
            let mutable goOn = true
            let colors = [|0..nVertices - 1|]
            let edges = this.OrdinalEdges

            while goOn do
                goOn <- false
                for edge in edges do
                    let i, j = edge
                    let colorI, colorJ = colors.[i], colors.[j]

                    if colorI < colorJ then
                        goOn <- true
                        colors.[j] <- colorI
                    elif colorJ < colorI then
                        goOn <- true
                        colors.[i] <- colorJ

            // normalize colors to run the range of [0..n - 1] (n = # of colors)
            let distinctColors = colors |> Array.distinct |> Array.sort |> Array.mapi (fun i value -> value, i) |> Map.ofArray
            colors
            |> Array.map (fun c -> distinctColors.[c])
        )

    let spanningTree =
        lazy(
                let edges = List<int * int>()
                let edgeNums = List<int>()
                
                // bfs traversal
                let visited = HashSet<int>()
                let queue = Queue<int>()
                queue.Enqueue 0
                visited.Add 0 |> ignore

                // optimization: if we have already visited all vertices - abort
                while queue.Count > 0 && visited.Count < nVertices do
                    let vertex = queue.Dequeue ()

                    let vertices, connectedEdges = getVerticesAndEdges vertex
                    let mutable n = 0
                    while n < vertices.Length && visited.Count < nVertices do
                        let v = vertices.[n]
                        let e = connectedEdges.[n]
                        if not (visited.Contains v) then
                            queue.Enqueue v
                            edges.Add(vertex, v)
                            edgeNums.Add(e)
                            visited.Add(v) |> ignore
                        n <- n + 1

                edges |> Seq.map(fun (st, e) -> verticesOrdinalToNames.[st], verticesOrdinalToNames.[e])
                |> HashSet
                , edgeNums.ToArray()
        )

    member this.NumVertices = nVertices
    member this.NumEdges = nEdges

    member this.Item
        with get vertex = ordinalFromName vertex |> getVertexConnections |> Array.map nameFromOrdinal

    member this.AsEnumerable = Seq.init nVertices (fun n -> nameFromOrdinal n, this.[nameFromOrdinal n])

    member this.Subgraph (vertices : 'a list) = Seq.init (vertices.Count()) (fun i -> vertices.[i], this.[vertices.[i]])

    member this.Reverse = reverse.Force()
    member this.RowIndex = rowIndex
    member this.ColIndex = colIndex

    member private this.GetConnectedVertices ordinal = getVertexConnections ordinal
    member private this.GetConnectedToMe ordinal = this.Reverse.GetConnectedVertices ordinal
    member private this.GetAllConnections ordinal =
        this.GetConnectedVertices ordinal |> fun g -> g.Union (this.GetConnectedToMe ordinal)

    member this.Connected vertex = vertex |> ordinalFromName |> this.GetConnectedVertices |> Array.map nameFromOrdinal
    member this.ConnectedToMe vertex = vertex |> ordinalFromName |> this.GetConnectedToMe |> Array.map nameFromOrdinal
    member this.AllConnected vertex = vertex |> ordinalFromName |> this.GetAllConnections |> Seq.map nameFromOrdinal |> Seq.toArray

    member this.NamedVertices = verticesNameToOrdinal
    /// <summary>
    /// Is this a Eulerian graph: i.e., in-degree of all vertices = out-degree
    /// </summary>
    member this.IsEulerian =
        this.IsConnected &&
            (this.Reverse.RowIndex = this.RowIndex || hasEulerPath.Force())

    /// <summary>
    /// Array of tuples of edge ordinals
    /// </summary>
    member this.OrdinalEdges =
        (lazy (
                [|0..nVertices - 1|]
                |> Array.map (fun v -> getVertexConnections v |> Array.map (fun c -> v, c))
                |> Array.concat
        )).Force()

    /// <summary>
    /// Array of tuples of all graph edges
    /// </summary>
    member this.Edges =
        this.OrdinalEdges
        |> Array.map (fun (start, end') -> verticesOrdinalToNames.[start], verticesOrdinalToNames.[end'])

    /// <summary>
    /// Finds all connected components and returns them as a list of vertex sets.
    /// </summary>
    member this.FindConnectedComponents () =
        this.Partition()
        |> Array.zip [|0..nVertices - 1|]
        |> Array.groupBy snd
        |> Array.map (fun (_, color) -> color |> Array.map (fun (v, c) -> verticesOrdinalToNames.[v]) )

    /// <summary>
    /// Finding the spanning tree by bfs traversal
    /// </summary>
    member this.SpanningTree = fst (spanningTree.Force())

    /// <summary>
    /// A boolean array the size of NumEdges, where "true" means
    /// this edge is part of the spanning tree.
    /// </summary>
    member this.SpanningTreeEdges = snd (spanningTree.Force())

    member this.IsConnected = this.Partition() |> Array.distinct |> Array.length |> ((=) 1)

    /// <summary>
    /// Find an Euler cycle or path
    /// </summary>
    /// <param name="start"></param>
    member this.FindEulerCycle (?start) =
        let mutable curVertex = defaultArg start 0

        let stack = Stack<int>()
        let visited = Dictionary<int, int []>()
        let start = curVertex
        let mutable cycle = []
        visited.Add(curVertex, this.GetConnectedVertices curVertex)
        let mutable first = true

        while stack.Count > 0 || first do
            first <- false
            let connected = visited.[curVertex]
            if connected.Length = 0 then
                cycle <- curVertex :: cycle
                curVertex <- stack.Pop()
            else
                stack.Push curVertex
                visited.[curVertex] <- connected.[1..]
                curVertex <- connected.[0]
                if not (visited.ContainsKey curVertex) then
                    visited.Add(curVertex, this.GetConnectedVertices curVertex)

        let path = start::cycle
        if path.Length <> this.NumEdges + 1 then []
        else
            start::cycle |> List.map (fun i -> verticesOrdinalToNames.[i])

    /// <summary>
    /// When we are not sure we are looking for a complete cycle
    /// </summary>
    member this.FindEulerPath () =
        // in the case of Eulerian path we need to figure out where to start:
        let curVertex =
            if hasEulerPath.Force() then
                let diffs = Array.map2 (-) this.RowIndex this.Reverse.RowIndex
                try // if the vertex with out-degre > in-degree comes first...
                    (diffs |> Array.findIndex (fun d -> d = 1)) - 1
                with
                _ -> diffs |> Array.findIndexBack (fun d -> d = -1)
            else
                0
        this.FindEulerCycle curVertex


    /// <summary>
    /// Edge-based partitioning of the graph into connected components
    /// </summary>
    member this.Partition () =
        if hasCuda.Force() && this.NumEdges >= 50000 then
            let dStart, dEnd = getEdgesGpu rowIndex colIndex
            partitionGpu dStart dEnd nVertices
            |> fun c -> c.Gather()
        else
            partitionLinear.Force()

    override this.Equals g2 =
        match g2 with
        | :? DirectedGraph<'a> as g ->
            if g.NumVertices = this.NumVertices then

                let grSort (gr : seq<'a * 'a[]>) =
                    gr |> Seq.sortBy fst

                let gseq = g.AsEnumerable |> grSort
                let thisSeq = this.AsEnumerable |> grSort

                let getKeys (gr : seq<'a * 'a[]>) =
                    gr |> Seq.map fst |> Seq.toArray

                let getVals (gr : seq<'a * 'a[]>) =
                    gr |> Seq.map (fun (a, b) -> b |> Array.sort) |> Seq.toArray

                let valuesEq (gr : seq<'a * 'a []>) (gr1 : seq<'a * 'a []>) =
                    not (Seq.zip (getVals gr) (getVals gr1) |> Seq.exists (fun (a, b) -> a <> b))

                let keysEq (gr : seq<'a * 'a []>) (gr1 : seq<'a * 'a []>) =
                    (getKeys gr) = (getKeys gr1)

                keysEq thisSeq gseq && valuesEq thisSeq gseq
            else false

        | _ -> false

    override this.GetHashCode() = this.AsEnumerable.GetHashCode()

type StrGraph = DirectedGraph<string>