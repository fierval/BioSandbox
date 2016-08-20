namespace GpuEuler

[<AutoOpen>]
module Euler =
    open Graphs
    open Alea.CUDA
    open Alea.CUDA.Utilities
    open Graphs.GpuGoodies

    let findEuler (gr : DirectedGraph<'a>) =
        let numEdges = gr.NumEdges

        // 1. find successors in the reverse graph notation
        let dStart, dEnd, dRevRowIndex = reverseGpu gr

        let edgeSucc, revRowIndex = successors dStart dRevRowIndex

        // 2. Partition the succesors graph
        // Create a line graph from the successor array:
        let linearGraph = StrGraph.FromVectorOfInts edgeSucc
        let partition = partitionLinear linearGraph.ColIndex

        // 3. Create GC graph, where each vertex is a partition of the
        // Successor linear graph
        let gcGraph, links, validity = generateCircuitGraph revRowIndex partition

        // 4. Create the spanning tree of the gcGraph & generate swips
        let dSwips = generateSwipsGpu gcGraph links numEdges

        // 5. Create the path by modifying the successor array
        let fixedSuccessors = successorSwaps dRevRowIndex dSwips validity
        fixedSuccessors


