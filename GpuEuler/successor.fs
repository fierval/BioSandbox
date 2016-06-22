module GpuEuler

open Graphs
open Alea.CUDA
open Alea.CUDA.Unbound
open Alea.CUDA.Utilities
open GpuSimpleSort
open Graphs.GpuGoodies
open System.Linq
open System.Collections.Generic

let worker = Worker.Default

[<Kernel;ReflectedDefinition>]
let groupSortedNums (ends : deviceptr<int>) len (grouped : deviceptr<int>) =
    let mutable idx = blockIdx.x * blockDim.x + threadIdx.x

    if idx <= len then
        if (idx <> 0 && idx < len && ends.[idx - 1] <> ends.[idx]) || idx = len then
            let mutable n = 0
            let cur = ends.[idx - 1]
            idx <- idx - 1
            while idx >= 0 && ends.[idx] = cur do
                idx <- idx - 1
                n <- n + 1
            ends.[idx] <- n
        else
            ends.[idx] <- 0

/// <summary>
/// Same thing as graph.Reverse, however, since we know we are dealing with
/// an Euler graph this function is much faster
/// </summary>
/// <param name="rowIndex"></param>
/// <param name="colIndex"></param>
let createReverseEuler (gr : StrGraph) =
    let start, end' = getEdgesGpu gr.RowIndex gr.ColIndex ||> sortStartEnd

    let revRowIndex =
        end'
        |> Array.groupBy id
        |> Array.map (fun (_, arr) -> arr.Length)
        |> Array.scan (+) 0

    StrGraph(revRowIndex, start, gr.NamedVertices)

