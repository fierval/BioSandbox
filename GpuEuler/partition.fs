namespace GpuEuler

    [<AutoOpen>]
    module Partition =
        
        open Graphs
        open Alea.CUDA
        open Alea.CUDA.Unbound
        open Alea.CUDA.Utilities
        open GpuSimpleSort
        open Graphs.GpuGoodies
        open System.Linq
        open System.Collections.Generic
        open System

        /// <summary>
        /// Kernel that does edge-based partitioning
        /// Algorithm 2 from http://www.massey.ac.nz/~dpplayne/Papers/cstn-089.pdf
        /// </summary>
        /// <param name="start">Edge starts</param>
        /// <param name="end'">Edge ends</param>
        /// <param name="colors">Colors</param>
        /// <param name="len">length of end' and start arrays - # of edges</param>
        /// <param name="stop">Should we continue?</param>
        [<Kernel; ReflectedDefinition>]
        let partionKernel (start : deviceptr<int>) (end': deviceptr<int>) (colors : deviceptr<int>) len (go : deviceptr<bool>) = 
            let idx = blockDim.x * blockIdx.x + threadIdx.x

            if idx < len then
                let i, j = start.[idx], end'.[idx]
                let colorI, colorJ = colors.[i], colors.[j]

                if colorJ < colorI then
                    go.[0] <- true
                    __atomic_min (colors + i) colorJ |> ignore

                elif colorI < colorJ then
                    go.[0] <- true
                    __atomic_min (colors + j) colorI |> ignore

        /// <summary>
        /// Run the partitioning kernel
        /// </summary>
        /// <param name="dStart">Starting points of edges</param>
        /// <param name="dEnd">Ending points of edges</param>
        /// <param name="nVertices">Num of graph vertices</param>
        let partitionGpu (dStart : DeviceMemory<int>) (dEnd: DeviceMemory<int>) nVertices =
            let lp = LaunchParam(divup dStart.Length blockSize, blockSize)

            let dColor = worker.Malloc([|0..nVertices - 1|])
            use dGo = worker.Malloc([|true|])

            while dGo.GatherScalar() do
                dGo.Scatter([|false|])
                worker.Launch <@ partionKernel @> lp dStart.Ptr dEnd.Ptr dColor.Ptr dStart.Length dGo.Ptr

            dColor