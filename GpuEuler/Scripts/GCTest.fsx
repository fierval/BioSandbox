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

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let grf = StrGraph.GenerateEulerGraph(100000, 5)

// reverse the graph to identify the row index of
// edges "coming in"
let dStart, dEnd, dRevRowIndex = reverseGpu grf

// find successor edges
let succ = successors dStart dRevRowIndex

// partition
let colors = partitionLinear succ

let sg = StrGraph.FromVectorOfInts succ

let rowIndex = dRevRowIndex.Gather()

// create CG graph
let gr, links = generateCircuitGraph rowIndex colors
//gr.Visualize()