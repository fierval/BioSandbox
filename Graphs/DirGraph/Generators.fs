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
module Extensions =
    type Graphs.DirectedGraph<'a when 'a:comparison> with
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
                            if st = connections.[i] then None // we have connected this vertex to all vertices possible
                            elif connectionsReverse.[idx] = 0 then Some(-1, (st, k + 1)) // we cannot connect to vertex idx: its inbound connections quota is met
                            else
                                connectionsReverse.[idx] <- connectionsReverse.[idx] - 1 // connect to this vertex and move on
                                Some(idx, (st + 1, k + 1)))
                    |> Seq.filter(fun x -> x >= 0)
                colIndex.AddRange cols // these are all the vertices we could connect the i'th vertex to.

            DirectedGraph(rowIndex, colIndex, [0..rowIndex.Count - 2].ToDictionary(string, id))

        static member GenerateEulerGraphAlt (vNum, eNum) =

            let rnd = Random(int DateTime.Now.Ticks)
            let edges =
                [
                    yield! [(0, 1); (1, 0)]
                    for i = 2 to vNum - 2 do
                        let a = rnd.Next(0, i)
                        let b = rnd.Next(0, i)
                        yield! [(a, i); (i, b); (b, a)]
                ].ToList()

            let curCount = edges.Count
            for i in [curCount + 1..2..eNum - 1] do
                let a = rnd.Next(0, vNum)
                let b = rnd.Next(0, vNum)
                edges.AddRange([a, b; b, a])


            let edgeArray = edges |> Seq.toArray |> Array.sortBy fst
            let colIndex = edgeArray |> Array.map snd
            let rowIndex =
                edgeArray
                |> Array.map fst
                |> Array.groupBy id
                |> Array.map (fun (k, v) -> v.Length)
                |> Array.scan (+) 0

            StrGraph(rowIndex, colIndex, [0..rowIndex.Count() - 2].ToDictionary(string, id))


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

        static member FromInts (rowIndex : int seq) (colIndex : int seq) =
            StrGraph(rowIndex, colIndex, [0..rowIndex.Count() - 2].ToDictionary(string, id))

        static member FromVectorOfInts (ints : int seq) =
            let rowIndex = [|0..ints.Count()|]
            StrGraph.FromInts rowIndex ints

        static member SaveStrs (gr : DirectedGraph<string>, fileName : string) =
            let toVertices (arr : string []) =
                if arr |> Array.isEmpty then String.Empty
                else
                    " -> " +
                    (arr
                    |> Array.reduce  (fun st e -> st + "," + string e))

            let strs =
                gr.AsEnumerable
                |> Seq.map (fun (v, arr) -> v + toVertices arr)
                |> Seq.toArray

            File.WriteAllLines(fileName, strs)

        /// <summary>
        /// Created "undirected" from "directed" graph by doubling up the edges
        /// </summary>
        /// <param name="gr"></param>
        //TODO: Doesn't handle vertices with now arcs
        static member CreateUndirected (gr : DirectedGraph<'a>) =
            let starts, colIndex =
                Array.concat [|gr.OrdinalEdges; gr.OrdinalEdges |> Array.map (fun (st, e) -> e, st)|]
                |> Array.sortBy fst
                |> Array.unzip

            let rowIndex =
                starts
                |> Array.groupBy id
                |> Array.map (fun (_, v) -> v.Length)
                |> Array.scan (+) 0

            DirectedGraph<'a>(rowIndex, colIndex, gr.NamedVertices)

        /// <summary>
        /// Generates a graph from the integer list of edges
        /// </summary>
        /// <param name="edges"></param>
        static member FromIntEdges (edges : seq<int * int>) =
            let adjecency =
                edges
                |> Seq.groupBy fst
                |> Seq.sortBy fst
                |> Seq.map (fun (key, sq) -> string key + "->" + (sq |> Seq.map snd |> Seq.fold (fun st e -> st + "," + string e) ""))

            StrGraph.FromStrings adjecency

        static member FromStrEdges (edges : seq<string * string>) =
            let adjecency =
                edges
                |> Seq.groupBy fst
                |> Seq.sortBy fst
                |> Seq.map (fun (key, sq) -> key + "->" + (sq |> Seq.map snd |> Seq.reduce (fun st e -> st + "," + e)))

            StrGraph.FromStrings adjecency
