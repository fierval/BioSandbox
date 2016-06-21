module GpuEuler

open Graphs
open Alea.CUDA
open Alea.CUDA.Utilities
open GpuSimpleSort

let blockSize = DirectedGraph.blockSize
let worker = DirectedGraph.worker
let target = DirectedGraph.target

let getEdgesGpu (gr : DirectedGraph<'a>) =
    let rowIndex = gr.RowIndex
    let colIndex = gr.ColIndex
    let len = rowIndex.Length

    let lp = LaunchParam(divup len blockSize, blockSize)

    let dStart = worker.Malloc<int>(colIndex.Length)
    let dEnd = worker.Malloc<int>(colIndex.Length)
    let dRowIndex = worker.Malloc(rowIndex)
    let dColIndex = worker.Malloc(colIndex)

    worker.Launch <@ DirectedGraph.toEdgesKernel @> lp dRowIndex.Ptr len dColIndex.Ptr dStart.Ptr dEnd.Ptr

    dStart, dEnd
