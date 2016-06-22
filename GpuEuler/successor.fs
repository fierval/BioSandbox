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

