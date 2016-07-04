namespace GpuEuler

    [<AutoOpen>]
    module GCEdges =
        open Graphs
        open Alea.CUDA
        open Alea.CUDA.Utilities
        open Graphs.GpuGoodies
        open System.Collections.Generic
        open GpuCompact

        let generateCircuitGraph (rowIndex : int []) (colors : int []) =
            let numColors = 1 + (colors |> Array.max)
            let numVertices = rowIndex.Length - 1
            let numEdges = rowIndex.[rowIndex.Length - 1]

            let status = Array.create numColors false
            let validity = Array.create numEdges true
            let ea = List<int>()
            let eb = List<int>()
            let links = List<int>()
            let mutable pre = 0

            for i = 0 to numVertices - 1 do
                for j = rowIndex.[i] to rowIndex.[i + 1] - 1 do
                    if not status.[colors.[j]] then
                        status.[colors.[j]] <- true
                    else
                        validity.[j] <- false

                pre <- rowIndex.[i]
                for j = rowIndex.[i] + 1 to rowIndex.[i + 1] - 1 do
                    if validity.[j] then
                        ea.Add colors.[j]
                        eb.Add colors.[pre]
                        links.Add j
                        pre <- j

                Array.fill status 0 status.Length false

            // convert to graph
            let eaArr = ea.ToArray()
            let ebArr = eb.ToArray()

            ea.AddRange ebArr
            eb.AddRange eaArr
            links.AddRange links


            let starts, colIndex, linkPos =
                Seq.zip3 ea eb links
                |> Seq.sortBy (fun (a, _, _) -> a)
                |> Seq.toArray
                |> Array.unzip3

            let newRowIndex =
                starts
                |> Array.groupBy id
                |> Array.map (fun (_, v) -> v.Length)
                |> Array.scan (+) 0

            (StrGraph.FromInts newRowIndex colIndex), linkPos