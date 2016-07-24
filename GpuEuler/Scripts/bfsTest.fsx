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


let euler = StrGraph.GenerateEulerGraph(8, 3, path=false)
let sptree = euler.SpanningTree
printfn "%A" (sptree.ToArray())
euler.Visualize(spanningTree=true)

let edges = bfs euler
