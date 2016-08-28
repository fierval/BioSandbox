namespace GpuEuler

[<AutoOpen>]
module Euler =
    open Graphs
    open System.Diagnostics

    let findEuler (gr : DirectedGraph<'a>) =
        let numEdges = gr.NumEdges

        // 1. find successors in the reverse graph notation
        let edgePredecessors = predecessors gr

        // 2. Partition the succesors graph
        // Create a line graph from the successor array:
        let partition, maxPartition = partitionLinear edgePredecessors

        if maxPartition <> 1 then
            // 3. Create GC graph, where each vertex is a partition of the
            // Successor linear graph
            let gcGraph, links, validity = generateCircuitGraph gr.RowIndex partition maxPartition

            // 4. Create the spanning tree of the gcGraph & generate swips
            let dSwips = generateSwipsGpu gcGraph links numEdges

            // 5. Create the path by modifying the successor array
            let fixedPredecessors = predecessorSwaps gr.RowIndex dSwips validity edgePredecessors
            fixedPredecessors
        else
            edgePredecessors



    let findEulerTimed (gr : DirectedGraph<'a>) =
        let numEdges = gr.NumEdges

        printfn "%s" (System.String.Format("Euler graph: vertices - {0:N}, edges - {1:N}", gr.NumVertices, gr.NumEdges))
        let gsw = Stopwatch()
        gsw.Start()

        let sw = Stopwatch()
        sw.Restart()
        let edgePredecessors = predecessors gr
        sw.Stop()

        printfn "1. Predecessors computed in %A" sw.Elapsed

        // 2. Partition the succesors graph
        // Create a line graph from the successor array:
        sw.Restart()
        let partition, maxPartition = partitionLinear edgePredecessors
        sw.Stop()

        printfn "2. Partitioned linear graph in %A" sw.Elapsed


        if maxPartition <> 1 then
            sw.Restart()
            // 3. Create GC graph, where each vertex is a partition of the
            // Successor linear graph
            let gcGraph, links, validity = generateCircuitGraph gr.RowIndex partition maxPartition
            sw.Stop()

            printfn "3. Circuit graph generated in %A" sw.Elapsed

            sw.Restart()
            // 4. Create the spanning tree of the gcGraph & generate swips
            let dSwips = generateSwipsGpu gcGraph links numEdges
            sw.Stop()

            printfn "4. Swips generated in %A" sw.Elapsed

            sw.Restart()
            // 5. Create the path by modifying the successor array
            let fixedPredecessors = predecessorSwaps gr.RowIndex dSwips validity edgePredecessors
            sw.Stop()

            printfn "5. Swips implemented in %A" sw.Elapsed
            gsw.Stop()
            printfn "GPU: Euler graph generated in %A" gsw.Elapsed
            fixedPredecessors
        else
            gsw.Stop()
            printfn "Euler graph generated in %A" gsw.Elapsed
            edgePredecessors

