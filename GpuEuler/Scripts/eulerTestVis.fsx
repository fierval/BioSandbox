#load "load-project-release.fsx"

open GpuEuler
open Graphs
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities
open System.Diagnostics
open System.Linq

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let N = 2
let k = 10
//let gr = StrGraph.GenerateEulerGraph(N, k)
let gr = StrGraph.GenerateEulerGraphAlt(N, N * k)

let numEdges = gr.NumEdges

// 1. find successors in the reverse graph notation
let rowIndex = gr.RowIndex

let edgePredecessors = predecessors gr

// 2. Partition the succesors graph
// Create a line graph from the successor array:
let linearGraph = StrGraph.FromVectorOfInts edgePredecessors
let partition, maxPartition = partitionLinear edgePredecessors
linearGraph.Visualize()

//if maxPartition <> 1 then
// 3. Create GC graph, where each vertex is a partition of the
// Successor linear graph
let gcGraph, links, validity = generateCircuitGraph rowIndex partition maxPartition
gcGraph.Visualize(spanningTree=true)

// 4. Create the spanning tree of the gcGraph & generate swips
//let dSwips = generateSwipsGpu gcGraph links numEdges

// 5. Create the path by modifying the successor array
//let fixedPredecessors = predecessorSwaps rowIndex dSwips validity edgeSucc
let fixedPredecessors = fixPredecessors gcGraph links edgePredecessors validity

let finalGraph = StrGraph.FromVectorOfInts fixedPredecessors
finalGraph.Reverse.Visualize()
gr.Visualize(edges=true)

