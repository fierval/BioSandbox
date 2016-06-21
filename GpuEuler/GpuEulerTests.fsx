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

type Marker =
    static member digr = graphGen 3 500 |> Arb.fromGen
    static member ``Edge Representation on GPU`` (gr : DirectedGraph<string>) =
        let start, end' = gr.Edges

        let dStart, dEnd = getEdgesGpu gr
        let startGpu, endGpu = dStart.Gather(), dEnd.Gather()
        start = startGpu && end' = endGpu

Arb.registerByType(typeof<Marker>)
Check.VerboseAll(typeof<Marker>)

