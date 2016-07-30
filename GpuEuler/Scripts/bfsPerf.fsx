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
let N = 15

sw.Restart()
let euler = StrGraph.GenerateEulerGraph(N, 3, path=false)
sw.Stop()
printfn "Generated euler graph in %A" sw.Elapsed
euler.Visualize(spanningTree = true, washNonSpanning = false)
euler.Visualize(spanningTree = true, washNonSpanning = true)

sw.Restart()
let sptree = euler.SpanningTree
let linearSpanTree = StrGraph.FromStrEdges sptree
linearSpanTree.Visualize()

printfn "Spanning tree generated in %A" sw.Elapsed

sw.Restart()
let edges = bfs euler
let sptreeGpu =
    (euler.Edges, edges)
    ||> Array.map2 (fun (s, e) incl -> (s, e, incl))
    |> Array.filter (fun (_, _, incl) -> incl)
    |> Array.map (fun (a, b, _) -> (a, b))

let gpuGenSpanTree = StrGraph.FromStrEdges sptreeGpu
gpuGenSpanTree.Visualize()

sw.Stop()
printfn "Spanning tree generated in %A" sw.Elapsed