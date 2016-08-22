namespace GpuEuler

[<AutoOpen>]
module Euler =
    open Graphs

    let findEuler (gr : DirectedGraph<'a>) =
        let numEdges = gr.NumEdges

        let dStart, dEnd, dRevRowIndex = reverseGpu gr
        let revRowIndex = dRevRowIndex.Gather()

        let edgeSucc = successors dStart dRevRowIndex

        // 2. Partition the succesors graph
        // Create a line graph from the successor array:
        let linearGraph = StrGraph.FromVectorOfInts edgeSucc
        let partition, maxPartition = partitionLinear linearGraph.ColIndex

        // special case: we got lucky & already have it!
        if maxPartition = 1 then
            edgeSucc
        else
            // 3. Create GC graph, where each vertex is a partition of the
            // Successor linear graph
            let gcGraph, links, validity = generateCircuitGraph revRowIndex partition
            gcGraph.Visualize(spanningTree=true)

            // 4. Create the spanning tree of the gcGraph & generate swips
            let dSwips = generateSwipsGpu gcGraph links numEdges

            // 5. Create the path by modifying the successor array
            let fixedSuccessors = successorSwaps dRevRowIndex dSwips validity edgeSucc
            fixedSuccessors


