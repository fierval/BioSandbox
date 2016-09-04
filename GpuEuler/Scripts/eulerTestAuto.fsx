#load "load-project-release.fsx"

open GpuEuler
open Graphs
open System.IO
open System

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let last = 20
let rnd = Random(int DateTime.UtcNow.Ticks)
let results : bool [] = Array.create last false

for i = 1 to last do
    let p = pown 2 i
    let n = rnd.Next(1, p)
    let k = rnd.Next(1, i)

    let gr = StrGraph.GenerateEulerGraph(n, k)
    printfn "Generated graph: vertices - %d, edges - %d" gr.NumVertices gr.NumEdges

    let edges = findEuler gr
    results.[i - 1] <- validate gr edges

    printfn "Cycle is %svalid" (if results.[i - 1] then "" else "not ")

let passed = Array.exists not results |> not

printfn "%s" (if passed then "PASSED!" else "FAILED!")
