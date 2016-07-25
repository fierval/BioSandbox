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

let sw = Stopwatch()
let N = 10 * 1024 * 1024

sw.Restart()
let euler = StrGraph.GenerateEulerGraph(N, 3, path=false)
sw.Stop()
printfn "Generated euler graph in %A" sw.Elapsed
//euler.Visualize(spanningTree = true)
sw.Restart()
let sptree = euler.SpanningTree
printfn "Spanning tree generated in %A" sw.Elapsed

sw.Restart()
let edges = bfs euler
//let sptreeGpu =
//    (euler.Edges, edges)
//    ||> Array.map2 (fun (s, e) incl -> (s, e, incl))
//    |> Array.filter (fun (_, _, incl) -> incl)
//    |> Array.map (fun (a, b, _) -> (a, b))

sw.Stop()
printfn "Spanning tree generated in %A" sw.Elapsed