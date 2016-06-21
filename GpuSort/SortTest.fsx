#load @"Scripts\load-project-debug.fsx"
#load "../packages/FSharp.Charting.0.90.14/FSharp.Charting.fsx"

open GpuSimpleSort
open Alea.CUDA
open Alea.CUDA.Unbound
open Alea.CUDA.Utilities
open System.IO
open System.Diagnostics
open FSharp.Charting
open System

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\release")

// load everything
let arr = generateRandomData 100000
sort arr
let start = 10000000

let seqElapsed, cudaElapsed =
    [1..10] 
    |> List.map
          (fun i -> 
            let arr = generateRandomData (start * i)
            let sw = Stopwatch()
            sw.Start()
            let carr = sort arr
            sw.Stop()
            let elapsedCuda = sw.ElapsedMilliseconds
            let elapsedCudaTs = sw.Elapsed

            sw.Restart()
            let sarr = Array.sort arr
            sw.Stop()

            printfn "Trial: %d, CUDA: %A, Sequential: %A" i elapsedCudaTs sw.Elapsed

            (i, float sw.ElapsedMilliseconds), (i, float elapsedCuda)
    )
    |> List.unzip

Chart.Combine(
    [Chart.Line(seqElapsed, Name="Sequential"); Chart.Line(cudaElapsed, Name="Alea.Cuda")])
    .WithYAxis(Log=false, Title = "msec")
    .WithXAxis(Title = String.Format("elements x {0:N0}", start))
    .WithLegend(InsideArea=false)
    
