module GpuDistinct

open Alea.CUDA
open Alea.CUDA.Utilities
open GpuCompact
open GpuSimpleSort
open System

[<Kernel;ReflectedDefinition>]
let distinctSortedNums (ends : deviceptr<int>) len (grouped : deviceptr<int>) =
    let mutable idx = blockIdx.x * blockDim.x + threadIdx.x

    if idx <= len then
        if (idx <> 0 && idx < len && ends.[idx - 1] <> ends.[idx]) || idx = len then
            let cur = ends.[idx - 1]
            idx <- idx - 1
            while idx >= 0 && ends.[idx] = cur do
                idx <- idx - 1
            grouped.[idx + 1] <- cur
        else
            grouped.[idx] <- 0

// kernel to use for compaction: the first element always gets in
[<Kernel; ReflectedDefinition>]
let createDistinctMap (arr : deviceptr<int>) len (out : deviceptr<int>) =
    let ind = blockIdx.x * blockDim.x + threadIdx.x

    if ind < len then
        out.[ind] <- if arr.[ind] <> 0 || ind = 0 then 1 else 0

/// <summary>
/// Disticnt
/// </summary>
/// <param name="dArr"></param>
let distinctGpu (dArr : DeviceMemory<int>) =
    use dSorted = sortGpu dArr
    use dGrouped = worker.Malloc<int>(dSorted.Length)

    let lp = LaunchParam(divup dSorted.Length blockSize, blockSize)
    worker.Launch <@ distinctSortedNums @> lp dSorted.Ptr dSorted.Length dGrouped.Ptr

    compactGpuWithKernel <@createDistinctMap @> dGrouped

let distinct (arr : int []) =
    use dArr = worker.Malloc(arr)
    let dDistinct = distinctGpu dArr
    dDistinct.Gather()