#load "load-project-debug.fsx"
open GpuEuler
open Graphs
open Alea.CUDA
open Alea.CUDA.Unbound
open DataGen
open System
open System.IO
open FsCheck
open GpuGoodies
open GpuCompact
open System.Diagnostics
open System

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

//let sparse = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"; "1 -> 2, 3"; "3 -> 4, 5"; "x -> y, z"; "2 -> 5"]
//let grs = StrGraph.FromStrings sparse


let gr = StrGraph.GenerateEulerGraph(20, 4)

let dStart, dEnd, dRevRowIndex = reverseGpu gr
let succ = successors dStart dRevRowIndex
let partition = (partitionLinear succ).Gather()

let lg = StrGraph.FromInts succ

gr.Visualize()
lg.Visualize()