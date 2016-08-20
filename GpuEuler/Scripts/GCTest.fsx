#load "load-project-debug.fsx"
open GpuEuler
open Graphs
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound
open DataGen
open System
open System.IO
open FsCheck
open GpuGoodies
open GpuCompact
open System.Diagnostics
open System
open DataGen
open GpuDistinct
open System.Linq
open System.Diagnostics

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let n = 15 * 1024 * 10
let k = 5
let sw = Stopwatch.StartNew()

let grf = StrGraph.GenerateEulerGraph(n, k)
sw.Stop()

sw.Restart()
// reverse the graph to identify the row index of
// edges "coming in"
let dStart, dEnd, dRevRowIndex = reverseGpu grf
sw.Stop()

sw.Restart()
// find successor edges
let succ, _ = successors dStart dRevRowIndex
sw.Stop()

// partition
sw.Restart()
let colors = partitionLinear succ
sw.Stop()
let sg = StrGraph.FromVectorOfInts succ

let rowIndex = dRevRowIndex.Gather()

// create CG graph
let gr, links, _ = generateCircuitGraph rowIndex colors
//gr.Visualize()

sw.Restart()
grf.FindEulerPath()
sw.Stop()