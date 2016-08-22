#load "load-project-debug.fsx"

open GpuEuler
open Graphs
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let gr = StrGraph.GenerateEulerGraph(12, 5)
gr.Reverse.Visualize(edges=true)

let numEdges = gr.NumEdges

// 1. find successors in the reverse graph notation
let dStart, dEnd, dRevRowIndex = reverseGpu gr
let revRowIndex = dRevRowIndex.Gather()

let edgeSucc = successors dStart dRevRowIndex
let origSucc : int [] = Array.zeroCreate edgeSucc.Length

edgeSucc.CopyTo(origSucc, 0)
// 2. Partition the succesors graph
// Create a line graph from the successor array:
let linearGraph = StrGraph.FromVectorOfInts edgeSucc
let partition, maxPartition = partitionLinear linearGraph.ColIndex
linearGraph.Visualize()

if maxPartition <> 1 then
    // 3. Create GC graph, where each vertex is a partition of the
    // Successor linear graph
    let gcGraph, links, validity = generateCircuitGraph revRowIndex partition
    gcGraph.Visualize(spanningTree=true)

    // 4. Create the spanning tree of the gcGraph & generate swips
    let dSwips = generateSwipsGpu gcGraph links numEdges
    let swips = dSwips.Gather()
    let edges = bfs gcGraph

    // 5. Create the path by modifying the successor array
    let fixedSuccessors = successorSwaps dRevRowIndex dSwips validity edgeSucc

    let finalGraph = StrGraph.FromVectorOfInts fixedSuccessors
    finalGraph.Visualize()
