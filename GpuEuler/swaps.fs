namespace GpuEuler

    [<AutoOpen>]
    module Swaps =
        open Graphs
        open Alea.CUDA
        open Alea.CUDA.Utilities
        open Graphs.GpuGoodies

        /// <summary>
        /// Fixes successors by swapping an edge identified by the spanning tree
        /// With the next "valid" edge. Parallelism by edges of a single vertex
        /// </summary>
        /// <param name="rowIndex">Row index of the original reversed graph</param>
        /// <param name="len">number of vertices</param>
        /// <param name="validity">validity array</param>
        /// <param name="swips">switching paris array</param>
        /// <param name="successors">original successors</param>
        [<Kernel; ReflectedDefinition>]
        let swapsKernel (rowIndex : deviceptr<int>) len (validity : deviceptr<bool>) (swips : deviceptr<int>) (predecessors : deviceptr<int>) =
            let idx = blockDim.x * blockIdx.x + threadIdx.x

            if idx < len - 1 then
                let end' = rowIndex.[idx + 1] - 1
                let start = rowIndex.[idx]
                for i = start to end' do
                    if swips.[i] > 0 then
                        let mutable j = i + 1
                        while j <= end' && not validity.[j] do
                            j <- j + 1

                        if j <= end' then
                            let temp = predecessors.[i]
                            predecessors.[i] <- predecessors.[j]
                            predecessors.[j] <- temp

        let predecessorSwaps (rowIndex : int []) (dSwips : DeviceMemory<int>) (validity : bool []) (predecessors : int[]) =
            use dRowIndex = worker.Malloc(rowIndex)
            let lp = LaunchParam(divup dRowIndex.Length blockSize, blockSize)

            use dPred = worker.Malloc(predecessors)
            use dValid = worker.Malloc(validity)

            worker.Launch <@swapsKernel@> lp dRowIndex.Ptr dRowIndex.Length dValid.Ptr dSwips.Ptr dPred.Ptr
            dPred.Gather()