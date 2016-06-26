#load "Scripts/load-project-debug.fsx"
open GpuEuler
open Graphs
open Alea.CUDA
open Alea.CUDA.Unbound
open DataGen
open System
open System.IO
open FsCheck
open GpuGoodies
open GpuCompact

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\release")

let gr = StrGraph.GenerateEulerGraph(20, 3)

let genEuler = 
    gen {
        return StrGraph.GenerateEulerGraph(5, 30)
}

type EulerCycle =
    static member digr = genEuler |> Arb.fromGen
    static member ``Reverse Euler Graph on GPU`` (gr : DirectedGraph<string>) =
        let _, _, dRevRowIndex = reverse gr
        let revRowIndex = dRevRowIndex.Gather()

        revRowIndex = gr.Reverse.RowIndex

Arb.registerByType(typeof<EulerCycle>)
Check.QuickAll(typeof<EulerCycle>)
