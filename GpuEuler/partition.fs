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
        open GpuDistinct

        [<Kernel; ReflectedDefinition>]
        let scatterDistinct (distinct : deviceptr<int>) len (map : deviceptr<int>) =
            let idx = blockDim.x * blockIdx.x + threadIdx.x
            if idx < len then
                map.[distinct.[idx]] <- idx

        [<Kernel; ReflectedDefinition>]
        let remapColors (colors : deviceptr<int>) len (map : deviceptr<int>) (compacted : deviceptr<int>) =
            let idx = blockDim.x * blockIdx.x + threadIdx.x
            if idx < len then
                compacted.[idx] <- map.[colors.[idx]]

        /// <summary>
        /// Normalizes our partition.
        /// Remaps partition colors to the range of 0..n - 1, n = # of colors.
        /// </summary>
        /// <param name="dColor"></param>
        let normalizePartition (dColor : DeviceMemory<int>) =

            // this compacts
            use dDistinct = distinctGpu dColor
            let maxPartition = dDistinct.GatherScalar(dDistinct.Length - 1) + 1

            use dMap = worker.Malloc<int>(maxPartition)
            let mutable lp = LaunchParam(divup dDistinct.Length blockSize, blockSize)

            worker.Launch <@scatterDistinct@> lp dDistinct.Ptr dDistinct.Length dMap.Ptr
            lp <- LaunchParam(divup dColor.Length blockSize, blockSize)

            let dCompacted = worker.Malloc<int>(dColor.Length)
            worker.Launch <@remapColors@> lp dColor.Ptr dColor.Length dMap.Ptr dCompacted.Ptr
            dCompacted.Gather(), maxPartition

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
        /// Kernel that does edge-based partitioning on a linear graph
        /// Algorithm 2 from http://www.massey.ac.nz/~dpplayne/Papers/cstn-089.pdf
        /// </summary>
        /// <param name="end'">Edge ends</param>
        /// <param name="colors">Colors</param>
        /// <param name="len">length of end' array - # of edges</param>
        /// <param name="stop">Should we continue?</param>
        [<Kernel; ReflectedDefinition>]
        let partionLinearGraphKernel (end': deviceptr<int>) (colors : deviceptr<int>) len (go : deviceptr<bool>) =
            let idx = blockDim.x * blockIdx.x + threadIdx.x

            if idx < len then
                let i, j = idx, end'.[idx]
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

            normalizePartition dColor

        /// <summary>
        /// Run the partitioning kernel
        /// </summary>
        /// <param name="dStart">Starting points of edges</param>
        /// <param name="dEnd">Ending points of edges</param>
        /// <param name="nVertices">Num of graph vertices</param>
        let partitionLinearGpu (end' : int []) =
            let lp = LaunchParam(divup end'.Length blockSize, blockSize)

            use dColor = worker.Malloc([|0..end'.Length - 1|])
            use dEnd = worker.Malloc(end')
            use dGo = worker.Malloc([|true|])

            while dGo.GatherScalar() do
                dGo.Scatter([|false|])
                worker.Launch <@ partionLinearGraphKernel @> lp dEnd.Ptr dColor.Ptr dEnd.Length dGo.Ptr

            normalizePartition dColor

        /// <summary>
        /// Partition the linear graph: graph, that consists only of cycles
        /// Where in-degree(v) = out-degree(v) = 1 for all v
        /// </summary>
        let partitionLinear (end' : int [])=
            let allVertices = HashSet<int>(end')
            let colors = Array.create end'.Length -1
            let mutable color = 0

            while allVertices.Count > 0 do
                let mutable v = allVertices.First()
                while colors.[v] < 0 do
                    allVertices.Remove v |> ignore
                    colors.[v] <- color
                    v <- end'.[v]
                color <- color + 1
            colors, color