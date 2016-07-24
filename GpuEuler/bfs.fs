namespace GpuEuler

    [<AutoOpen>]
    module BFS =
        open Graphs
        open Alea.CUDA
        open Alea.CUDA.Utilities
        open Graphs.GpuGoodies
        open System.Collections.Generic
        open GpuCompact

        [<Kernel; ReflectedDefinition>]
        let bfsKernel (front : deviceptr<bool>) len (frontOut : deviceptr<bool>) (visited : deviceptr<bool>) (level : deviceptr<int>) (count : deviceptr<int>)
            (edges : deviceptr<bool>) (rowIndex : deviceptr<int>) (colIndex : deviceptr<int>) (goOn : deviceptr<bool>) =

            let idx = blockIdx.x * blockDim.x + threadIdx.x
            if idx < len then
                frontOut.[idx] <- false
                if front.[idx] then
                    visited.[idx] <- true

                    for i = rowIndex.[idx] to rowIndex.[idx + 1] - 1 do
                        let vertex = colIndex.[i]
                        if not visited.[vertex] && level.[vertex] = 0 then
                            level.[vertex] <- level.[idx] + 1
                            __atomic_add (count + vertex) 1 |> ignore
                            frontOut.[vertex] <- true

                            edges.[i] <- true
                            goOn.[0] <- true

        /// <summary>
        /// Generates spanning tree by bfs on the gpu
        //  In order to use weak connectivity, need to generate the undirected graph first
        /// </summary>
        /// <param name="gr"></param>
        let bfs (gr : StrGraph) =

            let numEdges = gr.NumEdges
            let len = gr.NumVertices
            let lp = LaunchParam(divup len blockSize, blockSize)

            //let dStart, dEnd = getEdgesGpu gr.RowIndex gr.ColIndex // edges

            let allFalse = Array.create len false
            let allZero = Array.zeroCreate len

            use dRowIndex = worker.Malloc(gr.RowIndex)
            use dColIndex = worker.Malloc(gr.ColIndex)
            let dEdges = worker.Malloc(Array.create numEdges false)
            use goOn = worker.Malloc([|true|])
            use dVisted = worker.Malloc(allFalse)
            use front = worker.Malloc(allFalse)
            use frontOut = worker.Malloc(allFalse)
            use dLevel = worker.Malloc(allZero)
            use dCount = worker.Malloc(allZero)

            front.ScatterScalar true

            let getFront f = if f then front else frontOut

            let mutable flag = true

            while goOn.GatherScalar() do
                goOn.ScatterScalar(false)

                worker.Launch <@ bfsKernel @> lp (getFront flag).Ptr len (getFront (not flag)).Ptr dVisted.Ptr dLevel.Ptr dCount.Ptr dEdges.Ptr dRowIndex.Ptr dColIndex.Ptr goOn.Ptr
                flag <- not flag

            dEdges.Gather()