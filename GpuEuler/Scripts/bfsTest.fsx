#load "load-project-debug.fsx"

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
open System.Diagnostics
open System
open BFS
open System.Linq
open System.Collections.Generic

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let rnd = Random()
let N = 1 * 1024 * 10
for i = 1 to 10 do
    let euler = StrGraph.GenerateEulerGraph(N, 3, path=false)
    printfn "Test %d" i
    let edges = bfs euler
    let sptreeGpu =
        (euler.Edges, edges)
        ||> Array.map2 (fun (s, e) incl -> (s, e, incl))
        |> Array.filter (fun (_, _, incl) -> incl)
        |> Array.map (fun (a, b, _) -> (a, b))

    let isSpanningTree (gr : StrGraph) (sptree : seq<string * string>) =
        let sptree = sptree.ToArray()
        sptree.Length = gr.NumVertices - 1
        &&
        gr.Edges.Intersect(sptree).Count() = sptree.Length
        &&
        (
            let starts, ends = sptree |> Array.unzip
            starts.Union(ends).ToArray()
            |> Array.distinct
            |> Array.sort
            |> fun a -> a.Length = gr.NumVertices
        )

    let gr = euler
    let sptree = sptreeGpu

    if not (isSpanningTree euler sptreeGpu) then
        printfn "Failed!"
