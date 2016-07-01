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

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

//let sparse = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"; "1 -> 2, 3"; "3 -> 4, 5"; "x -> y, z"; "2 -> 5"]
//let grs = StrGraph.FromStrings sparse

let gener = graphGen 4 10

let gr = gener.Sample(500, 1).[0]

let sw = Stopwatch()
sw.Start()
let dStart, dEnd = getEdgesGpu gr.RowIndex gr.ColIndex
let color = partitionGpu dStart dEnd gr.NumVertices
let colors = color.Gather()
sw.Stop()
let eg = sw.Elapsed
sw.Restart()
let clin = gr.Partition()
sw.Stop()

clin = colors
printfn "%s" (String.Format("Vertices: {0:N0}, Edges: {1:N0}, CPU: {2}, GPU: {3}", gr.NumVertices, gr.NumEdges, sw.Elapsed, eg))

