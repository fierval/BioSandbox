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


let euler = StrGraph.GenerateEulerGraph(18, 4, path=false)

let edges, counts, levels = bfs euler
let sptreeGpu = 
    (euler.Edges, edges)
    ||> Array.map2 (fun (s, e) incl -> (s, e, incl))
    |> Array.filter (fun (_, _, incl) -> incl)
    |> Array.map (fun (a, b, _) -> (a, b))

euler.Visualize(spanningTree=true, washNonSpanning = true)

euler.SpanningTree <- HashSet sptreeGpu

euler.Visualize(spanningTree=true, washNonSpanning = true)

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
isSpanningTree euler sptreeGpu
