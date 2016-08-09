namespace GpuEuler

    [<AutoOpen>]
    module Swaps =
        open Graphs
        open Alea.CUDA
        open Alea.CUDA.Utilities
        open Graphs.GpuGoodies

        [<Kernel; ReflectedDefinition>]
        let swapsKernel (rowIndex : deviceptr<int>) len (validity : deviceptr<bool>) (swips : deviceptr<bool>) (successors : deviceptr<int>) =
            let idx = blockDim.x * blockIdx.x + threadIdx.x

            if idx < len - 1 then
                let end' = rowIndex.[idx + 1] - 1
                for i = rowIndex.[idx] to end' do
                    if swips.[idx] then
                        let mutable j = i + 1
                        while j <= end' && not validity.[j] do
                            j <- j + 1

                        if j <= end' then
                            let temp = successors.[i]
                            successors.[i] <- successors.[j]
                            successors.[j] <- temp

        let successorSwaps (dEnd : DeviceMemory<int>) (dSwips : DeviceMemory<bool>) (validity : bool []) (successors : int[]) =
            let lp = LaunchParam(divup dEnd.Length blockSize, blockSize)

            use dSucc = worker.Malloc(successors)
            use dValid = worker.Malloc(validity)

            worker.Launch <@swapsKernel@> lp dEnd.Ptr dEnd.Length dValid.Ptr dSwips.Ptr dSucc.Ptr
            dSucc.Gather()