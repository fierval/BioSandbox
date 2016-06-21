namespace Graphs

open System.Collections.Generic
open System.Linq
open System.IO
open System
open DrawGraph
open System.Linq
open Alea.CUDA

#nowarn "25"

[<AutoOpen>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DirectedGraph =

    let internal hasCuda = 
        lazy (
            try
                Device.Default.Name |> ignore
                true
            with
            _ ->    false
        )

    let gpuThresh = 1024 * 1024 * 10

    let getWorker () = if hasCuda.Force() then Some(Device.Default) else None

    let blockSize = 512
    let worker = Worker.Default
    let target = GPUModuleTarget.Worker worker


    // represent the graph as two arrays. For each vertex v, an edge is a tuple
    // start[v], end'[v]
    [<Kernel;ReflectedDefinition>]
    let toEdgesKernel (rowIndex : deviceptr<int>) len (colIndex : deviceptr<int>) (start : deviceptr<int>) (end' : deviceptr<int>) =
        let idx = blockIdx.x * blockDim.x + threadIdx.x
        if idx < len - 1 then
            for vertex = rowIndex.[idx] to rowIndex.[idx + 1] - 1 do
                start.[vertex] <- idx
                end'.[vertex] <- colIndex.[vertex]
        
    /// <summary>
    /// Instantiate a directed graph. Need number of vertices
    /// Format of the file:
    /// v -> a, b, c, d - v - unique vertex name for each line of the file. a, b, c, d - names of vertices it connects to.
    /// </summary>
    [<StructuredFormatDisplay("{AsEnumerable}")>]
    type DirectedGraph<'a when 'a:comparison> (rowIndex : int seq, colIndex : int seq, verticesNameToOrdinal : IDictionary<'a, int>) = 

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

        let getVertexConnections ordinal =
            let start = rowIndex.[ordinal]
            let end' = rowIndex.[ordinal + 1] - 1
            colIndex.[start..end']

        let asOrdinalsEnumerable () =
            Seq.init nVertices (fun i -> i, getVertexConnections i)

        let reverse =
            lazy (
                let allExistingRows = [0..rowIndex.Length - 1]                

                let grSeq = 
                    asOrdinalsEnumerable ()
                    |> Seq.map (fun (i, verts) -> verts |> Seq.map (fun v -> (v, i)))
                    |> Seq.collect id
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
        
        /// <summary>
        /// Create the graph from an array of strings
        /// </summary>
        /// <param name="lines">
        /// array of strings formatted: out_vertex -> in_v1, in_v2, in_v3,..
        ///</param>
        static member FromStrings (lines : string seq) =  

            let rowIndexRaw = List<int>()
            let colIndex = List<int>()

            let nameToOrdinal = Dictionary<string, int>() // vertices and the index to which they correspond

            let addVertex (line : string) =
                
                let vertex, connected = 
                    if line.Contains("->") then
                        line.Trim().Split([|"->"|], 2, StringSplitOptions.RemoveEmptyEntries) |> fun [|a; b|] -> a.Trim(), b.Trim()
                    else line.Trim(), ""

                let newVertex = not (nameToOrdinal.ContainsKey vertex)
                if newVertex then 
                    nameToOrdinal.Add(vertex, nameToOrdinal.Keys.Count)                    
                    rowIndexRaw.Add 0

                // get vertices connected to this one
                let connectedVertices =
                    if not (String.IsNullOrEmpty connected) then
                        connected.Split([|','|], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.map (fun e -> e.Trim())
                    else [||]

                let ordinal = nameToOrdinal.[vertex]

                // store all the vertices we have not seen before
                let newVertices = connectedVertices.Except nameToOrdinal.Keys |> Seq.toArray
                newVertices
                |> Seq.iter (fun v -> nameToOrdinal.Add(v, nameToOrdinal.Keys.Count))
                
                // extend the new rows
                let newVerticesZeroes = Array.zeroCreate newVertices.Length
                rowIndexRaw.AddRange newVerticesZeroes

                // for now we will store the number of vertices in the row index
                // entry for the given row. We will need to scan it and update its values later
                rowIndexRaw.[nameToOrdinal.[vertex]] <- connectedVertices.Length

                let connectedOrdinals = 
                    connectedVertices
                    |> Seq.map (fun v -> nameToOrdinal.[v])

                // if we are inserting a "retoractive" row, we need to know where we are inserting it!
                if newVertex then colIndex.AddRange connectedOrdinals
                else
                    let rowIndexCur = rowIndexRaw |> Seq.scan (+) 0 |> Seq.toList
                    colIndex.InsertRange(rowIndexCur.[ordinal], connectedOrdinals)

            lines |> Seq.iter addVertex

            DirectedGraph<string>(rowIndexRaw |> Seq.scan (+) 0, colIndex, nameToOrdinal)


        /// <summary>
        /// Create the graph from a file
        /// </summary>
        /// <param name="fileName"></param>
        static member FromFile (fileName : string) =

            if String.IsNullOrWhiteSpace fileName || not (File.Exists fileName) then failwith "Invalid file"
            
            let lines = File.ReadLines(fileName)
            DirectedGraph<string>.FromStrings(lines)                

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

        /// <summary>
        /// Is this a Eulerian graph: i.e., in-degree of all vertices = out-degree
        /// </summary>
        member this.IsEulerian = 
            this.IsConnected && 
                (this.Reverse.RowIndex = this.RowIndex || hasEulerPath.Force())
        
        member this.Edges =
            [|0..nVertices - 1|]
            |> Array.mapi (fun i v -> getVertexConnections v |> Array.map (fun c -> i, c))
            |> Array.concat
            |> Array.unzip
             
        /// <summary>
        /// Generates a Eulerian graph
        /// </summary>
        /// <param name="n"> number of vertices </param>
        /// <param name="k"> max number of connections in one direction</param>
        static member GenerateEulerGraph(n, k, ?path) =
            let isPath = defaultArg path false

            let rnd = Random(int DateTime.Now.Ticks)
            let connections = Array.init n (fun i -> rnd.Next(1, k + 1))
            let connectionsReverse = Array.zeroCreate n
            connections.CopyTo(connectionsReverse, 0)
            
            if isPath then
                let outLessThanIn = rnd.Next(n)
                let mutable inLessThanOut = rnd.Next(n)
                while outLessThanIn = inLessThanOut do
                    inLessThanOut <- rnd.Next(n)
                connections.[outLessThanIn] <- connections.[outLessThanIn] - 1
                connectionsReverse.[inLessThanOut] <- connections.[inLessThanOut] - 1

            let rowIndex = [0].ToList()
            let colIndex = List<int>()
            for i = 0 to n - 1 do
                rowIndex.Add(0)
                rowIndex.[i + 1] <- rowIndex.[i] + connections.[i]
                
                // scan vertices starting from vertex i and grab the next available vertex to connect to while it is possible 
                // connectionsReverse keeps track of each vertex ability to serve as an inbound vertex. At the end, all
                // of its elements should be eq to 0
                let cols = 
                    (0, 1)
                    |> Seq.unfold 
                        (fun (st, k) -> 
                            let idx = (i + k) % n
                            if st = connections.[i] then None 
                            elif connectionsReverse.[idx] = 0 
                            then Some(-1, (st, k + 1))
                            else
                                connectionsReverse.[idx] <- connectionsReverse.[idx] - 1
                                Some(idx, (st + 1, k + 1)))
                    |> Seq.filter(fun x -> x >= 0)
                colIndex.AddRange cols    
            DirectedGraph(rowIndex, colIndex, ([0..rowIndex.Count - 2].ToDictionary((fun s -> s.ToString()), id) :> IDictionary<string, int>))
        
        /// <summary>
        /// Finds all connected components and returns them as a list of vertex sets.
        /// </summary>
        member this.FindConnectedComponents (?oneOnly) =
            let oneOnly = defaultArg oneOnly false

            let rnd = Random()
            let mutable vertices = Enumerable.Range(0, this.NumVertices) |> Seq.toList
            
            [
                while vertices.Count() > 0 do
                    let idx = vertices.[rnd.Next(vertices.Count())]
                    
                    let connected = 
                        List.unfold 
                            (fun (prevVertices : HashSet<int>, visited) -> 
                                let visitVerts = 
                                    visited 
                                    |> Array.map (this.GetAllConnections >> Seq.toArray) 
                                    |> Array.concat |> Array.distinct
                                    |> fun a -> a.Except prevVertices |> Seq.toArray
                                let prevLen = prevVertices.Count                                        
                                visitVerts |> Array.iter (prevVertices.Add >> ignore)
                                if prevVertices.Count = prevLen then None
                                else Some(prevVertices, (prevVertices, visitVerts))
                            ) (HashSet<int>([idx]), [|idx|]) 
                        |> List.take 1
                        |> List.exactlyOne

                    vertices <- if oneOnly then [] else vertices.Except connected |> Seq.toList

                    yield connected |> Seq.map (fun i -> verticesOrdinalToNames.[i]) |> Seq.toList
            ]
            
        // GPU worker                                                    
        member private this.Worker = getWorker()

        member this.IsConnected = 
            this.FindConnectedComponents (oneOnly = true) 
            |> List.exactlyOne 
            |> fun l -> l.Length = verticesNameToOrdinal.Count
      
        member this.FindEulerPath () =
            if not this.IsEulerian then []
            else
                // in the case of Eulerian path we need to figure out where to start:
                let mutable curVertex = 
                    if hasEulerPath.Force() then
                        let diffs = Array.map2 (-) this.RowIndex this.Reverse.RowIndex
                        try // if the vertex with out-degre > in-degree comes first...
                            (diffs |> Array.findIndex (fun d -> d = 1)) - 1
                        with
                        _ -> diffs |> Array.findIndexBack (fun d -> d = -1)
                    else
                        0    

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

                start::cycle |> List.map (fun i -> verticesOrdinalToNames.[i])
                        
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
                    