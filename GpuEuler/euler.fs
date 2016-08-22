namespace GpuEuler

[<AutoOpen>]
module Euler =
    open Graphs

    let findEuler (gr : DirectedGraph<'a>) =
        let numEdges = gr.NumEdges

        // 1. find successors in the reverse graph notation
        let edgePredecessors = predecessors gr

        // 2. Partition the succesors graph
        // Create a line graph from the successor array:
        let linearGraph = StrGraph.FromVectorOfInts edgePredecessors
        let partition, maxPartition = partitionLinear linearGraph.ColIndex
        linearGraph.Visualize()

        if maxPartition <> 1 then
            // 3. Create GC graph, where each vertex is a partition of the
            // Successor linear graph
            let gcGraph, links, validity = generateCircuitGraph gr.RowIndex partition

            // 4. Create the spanning tree of the gcGraph & generate swips
            let dSwips = generateSwipsGpu gcGraph links numEdges

            // 5. Create the path by modifying the successor array
            let fixedPredecessors = predecessorSwaps gr.RowIndex dSwips validity edgePredecessors
            fixedPredecessors
        else
            edgePredecessors


