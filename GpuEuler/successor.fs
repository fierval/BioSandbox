module GpuEuler

open Graphs
open Alea.CUDA
open Alea.CUDA.Unbound
open Alea.CUDA.Utilities
open GpuSimpleSort
open Graphs.GpuGoodies
open System.Linq
open System.Collections.Generic
open System
open GpuCompact

let worker = Worker.Default

/// <summary>
/// Groups counts of values in a sorted array together:
/// 1 1 1 2 2 2 2 3 3... -> 3 0 0 4 0 0 0 2 ...
/// </summary>
/// <param name="ends"></param>
/// <param name="len"></param>
/// <param name="grouped"></param>
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
            grouped.[idx + 1] <- n
        else
            grouped.[idx] <- 0

/// <summary>
/// Assigns successors of all edges:
/// 1 -> 2 is a successor of edge 3 -> 1
/// </summary>
/// <param name="starts">Array of edges starts</param>
/// <param name="len">num of vertices</param>
/// <param name="curPointers">"rowIndex" array of ending vertices</param>
/// <param name="successors">array of successors</param>
[<Kernel; ReflectedDefinition>]
let assignSuccessors (starts : deviceptr<int>) len (curPointers : deviceptr<int>) (successors : deviceptr<int>) =
    let mutable idx = blockIdx.x * blockDim.x + threadIdx.x
    if idx <= len then
        successors.[curPointers.[starts.[idx]]] <- idx
        __atomic_add (curPointers + starts.[idx]) 1 |> ignore

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

let getRevRowIndex (dEnd : DeviceMemory<int>) =
    let len = dEnd.Length
    let lp = LaunchParam (divup len blockSize, blockSize)

    let dGrouped = worker.Malloc(Array.zeroCreate len)

    worker.Launch <@ groupSortedNums @> lp dEnd.Ptr len dGrouped.Ptr
    dGrouped

/// <summary>
/// Given an Eulerian graph - create start, end and rowIndex arrays
/// The "end" array will be the colIndex array for the reversed graph
/// </summary>
/// <param name="gr"></param>
let reverse (gr : StrGraph) =
    if not gr.IsEulerian then failwith "Not Eulerian"

    let dStart, dEnd = 
        getEdgesGpu gr.RowIndex gr.ColIndex 
        ||> sortStartEndGpu

    // Get the row index of the reverse graph
    let dGrouped = getRevRowIndex dEnd
    let dCompacted = compactGpu dGrouped

    // Row index of the reversed graph will be a scan of the compacted array
    use scanModule = new DeviceScanModule<int>(GPUModuleTarget.Worker(worker), <@ (+) @>)
    use scanner = scanModule.Create(dCompacted.Length)

    let dRevRowIndex = worker.Malloc(dCompacted.Length + 1)
    dRevRowIndex.ScatterScalar(0)
    scanner.InclusiveScan(dCompacted.Ptr, dRevRowIndex.Ptr + 1, dCompacted.Length)

    dStart, dEnd, dRevRowIndex
