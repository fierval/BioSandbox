#load "Scripts/load-project-debug.fsx"
open GpuEuler
open Graphs
open Alea.CUDA
open Alea.CUDA.Unbound
open DataGen
open System
open System.IO
open FsCheck

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\release")

let gr = StrGraph.GenerateEulerGraph(20, 3)

type EulerCycle =
    static member digr = graphGen 3 100 |> Arb.fromGen
    static member ``Edge Representation on GPU`` (gr : DirectedGraph<string>) =
        let start, end' = gr.Edges

        let dStart, dEnd = getEdgesGpu gr
        let startGpu, endGpu = dStart.Gather(), dEnd.Gather()
        start = startGpu && end' = endGpu

    static member ``Sort ordinal end edges on GPU`` (gr : DirectedGraph<string>) =
        let startCpu, endCpu =
            gr.Edges
            ||> Array.zip
            |> Array.sortBy (fun (x, y) -> y, x)
            |> Array.unzip

        let startGpu, endGpu =
            let dStart, dEnd = getEdgesGpu gr
            sortStartEnd dStart dEnd gr.NumEdges

        startCpu = startGpu && endCpu = endGpu

Arb.registerByType(typeof<EulerCycle>)
Check.QuickAll(typeof<EulerCycle>)

let digr = graphGen 3 100
let graphs = digr.Sample(3, 100)

