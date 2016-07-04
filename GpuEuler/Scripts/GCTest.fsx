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

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let grf = StrGraph.GenerateEulerGraph(230, 5)

let dStart, dEnd, dRevRowIndex = reverseGpu grf
let succ = successors dStart dRevRowIndex
let dColors = partitionLinear succ

let partitoined = StrGraph.FromVectorOfInts succ
let rowIndex = dRevRowIndex.Gather()
let colors = dColors.Gather()

let gr, links = generateCircuitGraph rowIndex colors
gr.Visualize()