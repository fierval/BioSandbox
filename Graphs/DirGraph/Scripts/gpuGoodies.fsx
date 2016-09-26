#load "load-project-release.fsx"
#r "System.Configuration"
#load @"..\..\..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#r @"..\..\..\GpuCompact\bin\Release\GpuCompact.dll"

open Graphs
open GpuGoodies
open GpuCompact
open System.IO
open FsCheck
open System

open System.Diagnostics
let sw = Stopwatch()

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\..\release")

// suppress GPU application
let mutable N = 10 * 1024 * 1024
let k = 5

sw.Restart()
let gr = StrGraph.GenerateEulerGraph(N, k)
sw.Stop()

printfn "Graph: %s vertices, %s edges generated in %A" (String.Format("{0:N0}", gr.NumVertices)) (String.Format("{0:N0}", gr.NumEdges)) sw.Elapsed

sw.Restart()
let starts, ends = getEdges gr.RowIndex gr.ColIndex

sw.Stop()
printfn "GPU edges: %A" sw.Elapsed

sw.Restart()

gr.OrdinalEdges

sw.Stop()
printfn "CPU edges: %A" sw.Elapsed
