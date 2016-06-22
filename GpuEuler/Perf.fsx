#load "Scripts/load-project-debug.fsx"

open Graphs
open Alea.CUDA
open Alea.CUDA.Utilities
open System.IO
open System.Diagnostics
open GpuEuler
open System
open GpuGoodies

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\release")

let sw = Stopwatch()
let N = 15000000
sw.Start()

let gr = StrGraph.GenerateEulerGraph(N, 5)
sw.Stop()

printfn "%s" (String.Format("Generated {0:N0} vertices in {1}", N, sw.Elapsed))

sw.Restart()

let revgr = gr.Reverse

sw.Stop()

printfn "Elapsed to reverse: %A" sw.Elapsed

let gr1 = StrGraph.GenerateEulerGraph(20, 3)

let dStart, dEnd = getEdgesGpu gr1.RowIndex gr1.ColIndex
sortStartEnd dStart dEnd

sw.Restart()
let revgr1 = createReverseEuler gr
sw.Stop()

printfn "Elapsed GPU: %A" sw.Elapsed




