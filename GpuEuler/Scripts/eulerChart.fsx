#load "load-project-release.fsx"
#load @"..\..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"

#load "load-project-release.fsx"

open GpuEuler
open Graphs
open System.IO
open System.Diagnostics
open FSharp.Charting

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let sw = Stopwatch()

// Warm up the GPU
findEuler <| StrGraph.GenerateEulerGraph(8, 5)

let N = 1024 * 1024
let k = 5
let avgedges = [1..k] |> List.map float |> List.average


let mutable cpu : (int * float) list = []
let mutable gpu : (int * float) list = []

for i in [0..5..20] do
    let n = (i + (if i = 0 then 1 else 0)) * N
    printfn "%s" (System.String.Format("Generating euler graph: vertices = {0:N0}; avg out/vertex: {1:N0}", n, avgedges))

    sw.Restart()
    let gr = StrGraph.GenerateEulerGraph(n, k)
    sw.Stop()
    printfn "Generated euler graph in %A" sw.Elapsed

    sw.Restart()
    findEuler gr |> ignore
    sw.Stop()
    printfn "GPU: %A" sw.Elapsed
    gpu <- (gr.NumEdges, float sw.Elapsed.Seconds) :: gpu

    sw.Restart()
    gr.FindEulerCycle() |> ignore
    sw.Stop()
    printfn "CPU: %A" sw.Elapsed
    cpu <- (gr.NumEdges, float sw.Elapsed.Seconds) :: cpu

Chart.Combine(
    [Chart.Line(cpu, Name="CPU");
        Chart.Line(gpu, Name="GPU")
    ])
    .WithYAxis(Log=false, Title = "sec")
    .WithXAxis(Title = "edges", Min= float N)
    .WithLegend(InsideArea=false)

