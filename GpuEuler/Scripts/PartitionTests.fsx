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

//let sparse = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"; "1 -> 2, 3"; "3 -> 4, 5"; "x -> y, z"; "2 -> 5"]
//let grs = StrGraph.FromStrings sparse


//let gr = StrGraph.GenerateEulerGraph(120, 4)
//
//let dStart, dEnd, dRevRowIndex = reverseGpu gr
//let succ = successors dStart dRevRowIndex
//let partition = (partitionLinear succ).Gather()

let genr = graphGen 4 5

let grf = genr.Sample(15, 1).[0]
let dStart, dEnd = getEdgesGpu grf.RowIndex grf.ColIndex
let dColor = partitionGpu dStart dEnd grf.NumVertices

let colors = grf.Partition()

let distinctColors = colors |> Array.distinct |> Array.sort |> Array.mapi (fun i value -> value, i) |> Map.ofArray

let modified =
    colors
    |> Array.map (fun c -> distinctColors.[c])

let lengths = modified |> Array.groupBy id |> Array.map (fun (_, vals) -> vals.Length)

let actual = dColor |> fun r -> r.Gather() |> Array.groupBy id |> Array.map (fun (_, vals) -> vals.Length)

