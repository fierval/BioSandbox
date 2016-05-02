namespace Graphs
open System.Collections.Generic
open System.Linq
open System.IO
open System
open DrawGraph

#nowarn "25"

[<AutoOpen>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DirectedGraph =

    /// <summary>
    /// Instantiate a directed graph. Need number of vertices
    /// Format of the file:
    /// v -> a, b, c, d - v - unique vertex name for each line of the file. a, b, c, d - names of vertices it connects to.
    /// </summary>
    type DirectedGraph (nVertex : int, rowIndex : int seq, colIndex : int seq, verticesNameToOrdinal : IDictionary<string, int>, ?isCSR : bool) = 

        let mutable isCSR = defaultArg isCSR true
        let rowIndex  = rowIndex.ToArray()
        let colIndex = colIndex.ToArray()
        let mutable nnz = rowIndex.[rowIndex.Length - 1]

        let ordinalToNames () =
            let res : string [] = Array.zeroCreate verticesNameToOrdinal.Count
            verticesNameToOrdinal 
            |> Seq.iter (fun kvp -> res.[kvp.Value] <- kvp.Key)

            res

        let rowSize = nVertex
        let verticesNameToOrdinal = verticesNameToOrdinal
        let verticesOrdinalToNames = ordinalToNames()

        let nameFromOrdinal = fun ordinal -> verticesOrdinalToNames.[ordinal]
        let ordinalFromName = fun name -> verticesNameToOrdinal.[name]

        let getVertexConnections ordinal =
            let start = rowIndex.[ordinal]
            let end' = rowIndex.[ordinal + 1] - 1
            colIndex.[start..end']


        /// <summary>
        /// Create the graph from an array of strings
        /// </summary>
        /// <param name="lines"></param>
        /// <param name="isCSR"></param>
        static member FromStrings (lines : string seq, ?isCSR : bool) =  

            let isCSR = defaultArg isCSR true
            let rowIndexRaw = List<int>()
            let colIndex = List<int>()
            let vertices = List<string>()            
            let nameToOrdinal = Dictionary<string, int>() // vertices and the index to which they correspond

            let addVertex (line : string) =
                
                let vertex, connected = 
                    if line.Contains("->") then
                        line.Trim().Split([|"->"|], 2, StringSplitOptions.RemoveEmptyEntries) |> fun [|a; b|] -> a.Trim(), b.Trim()
                    else line.Trim(), ""

                if not (nameToOrdinal.ContainsKey vertex) then 
                    nameToOrdinal.Add(vertex, nameToOrdinal.Keys.Count)                    
                    rowIndexRaw.Add 0

                // get vertices connected to this one
                let connectedVertices =
                    if not (String.IsNullOrEmpty connected) then
                        connected.Split([|','|], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.map (fun e -> e.Trim())
                    else [||]


                // store all the vertices we have not seen before
                let newVertices = connectedVertices.Except nameToOrdinal.Keys |> Seq.toArray
                newVertices
                |> Seq.iter (fun v -> nameToOrdinal.Add(v, nameToOrdinal.Keys.Count))
                
                // extend the new rows
                rowIndexRaw.AddRange (Array.zeroCreate newVertices.Length)


                // for now we will store the number of vertices in the row index
                // entry for the given row. We will need to scan it and update its values later
                rowIndexRaw.[nameToOrdinal.[vertex]] <- connectedVertices.Length

                connectedVertices
                |> Seq.map (fun v -> nameToOrdinal.[v])
                |> colIndex.AddRange

            lines |> Seq.iter addVertex

            DirectedGraph(rowIndexRaw.Count, rowIndexRaw |> Seq.scan (+) 0, colIndex, nameToOrdinal, isCSR)


        /// <summary>
        /// Create the graph from a file
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="isCSR"></param>
        static member FromFile (fileName : string, ?isCSR : bool) =
            let isCSR = defaultArg isCSR true

            if String.IsNullOrWhiteSpace fileName || not (File.Exists fileName) then failwith "Invalid file"
            
            let lines = File.ReadLines(fileName)
            DirectedGraph.FromStrings(lines, isCSR)                

        member this.Vertices = nVertex
        member this.Item
            with get vertex = ordinalFromName vertex |> getVertexConnections |> Array.map nameFromOrdinal

        member this.AsEnumerable = Seq.init nVertex (fun n -> nameFromOrdinal n, this.[nameFromOrdinal n])

        member this.Visualize() =
            let self = this.AsEnumerable

            let visualizable = 
                self 
                |> Seq.map 
                    (fun (v, c) -> 
                        if c.Length = 0 then v
                        else
                        c |> Seq.map (fun s -> v + (if s.Length = 0 then "" else " -> " + s)) |> Seq.reduce (fun acc e -> acc + "; " + e))
                |> Seq.reduce (fun acc e -> acc + "; " + e)
                |> fun v -> "digraph {" + v + "}"

            createGraph visualizable None

            