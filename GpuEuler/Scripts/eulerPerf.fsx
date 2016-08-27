#load "load-project-debug.fsx"

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

let N = 10 * 1024 * 1024
let k = 5

printfn "Generating euler graph: %d, %d" N, k
sw.Restart()
let gr = StrGraph.GenerateEulerGraph(N, k)
sw.Stop()
printfn "Generated euler graph in %A" sw.Elapsed

sw.Restart()
let eulerCycle = findEuler gr
sw.Stop()

printfn "GPU: %A" sw.Elapsed

sw.Restart()
let eulerVert = gr.FindEulerPath()
sw.Stop()

printfn "CPU: %A" sw.Elapsed

