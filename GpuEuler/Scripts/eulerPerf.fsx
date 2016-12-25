#load "load-project-release.fsx"
#load "../../packages/FSharp.Charting.0.90.14/FSharp.Charting.fsx"

open GpuEuler
open Graphs
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities
open System.Diagnostics
open FSharp.Charting
open System

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")
let sw = Stopwatch()

// Warm up the GPU
findEuler <| StrGraph.GenerateEulerGraph(8, 5)

let N = 1024 * 1024
let k = 7
let avgedges k = [1..k] |> List.map float |> List.average

let seqElapsed, cudaElapsed =
    [1..10]
    |> List.map
        (fun i ->
                printfn "%s" (System.String.Format("Generating euler graph: vertices = {0:N0}; avg out/vertex: {1:N0}", N * i, avgedges k))
                sw.Restart()
                let gr = StrGraph.GenerateEulerGraph(N * i, k)
                sw.Stop()
                printfn "Generated euler graph in %A, edges: %s" sw.Elapsed (String.Format("{0:N0}", gr.NumEdges))
                sw.Restart()
                let eulerCycle = findEulerTimed gr
                sw.Stop()
                let cuda = float sw.ElapsedMilliseconds

                sw.Restart()
                let eulerVert = gr.FindEulerCycle()
                sw.Stop()
                let cpu = float sw.ElapsedMilliseconds

                printfn "CPU: Euler cycle generated in %A" sw.Elapsed
                (float gr.NumEdges / float N, cpu), (float gr.NumEdges / float N, cuda)
        )
    |> List.unzip

let cpu, gpu =
    (seqElapsed, cudaElapsed)
    ||> List.map2 (fun (e1, t1) (e2, t2) -> (String.Format("{0:N0}", e1 * float N / 1000000.), t1 / 1000.0), (String.Format("{0:N0}", e2 * float N / 1000000.0), t2 / 1000.0))
    |> List.unzip

Chart.Combine(
    [Chart.Line(cpu, Name="CPU"); Chart.Line(gpu, Name="Alea.Cuda")])
    .WithYAxis(Log=false, Title = "sec")
    .WithXAxis(Title = String.Format("edges x {0:N0}", 1000000))
    .WithLegend(InsideArea=false)
