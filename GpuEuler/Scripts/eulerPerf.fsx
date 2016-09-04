﻿#load "load-project-release.fsx"

open GpuEuler
open Graphs
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities
open System.Diagnostics

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")
let sw = Stopwatch()

// Warm up the GPU
findEuler <| StrGraph.GenerateEulerGraph(8, 5)

let N = 20 * 1024 * 1024
let k = 5
let avgedges = [1..k] |> List.map float |> List.average

printfn "%s" (System.String.Format("Generating euler graph: vertices = {0:N0}; avg out/vertex: {1:N0}", N, avgedges))

sw.Restart()
let gr = StrGraph.GenerateEulerGraph(N, k)
sw.Stop()
printfn "Generated euler graph in %A" sw.Elapsed

let eulerCycle = findEulerTimed gr

sw.Restart()
let eulerVert = gr.FindEulerCycle()
sw.Stop()

printfn "CPU: Euler graph generated in %A" sw.Elapsed
