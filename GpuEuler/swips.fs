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
        let bfsKernel (front : deviceptr<bool>) len (frontOut : deviceptr<bool>) (visited : deviceptr<bool>) (level : deviceptr<int>) (edges : deviceptr<bool>) (rowIndex : deviceptr<int>) (colIndex : deviceptr<int>) (goOn : deviceptr<bool>) =

            let idx = blockIdx.x * blockDim.x + threadIdx.x
            if idx < len then
                frontOut.[idx] <- false
                if front.[idx] then
                    visited.[idx] <- true

                    for i = rowIndex.[idx] to rowIndex.[idx + 1] - 1 do
                        let vertex = colIndex.[i]
                        if not visited.[vertex] then
                            let oldLevel = __atomic_exch (level + vertex) 1
                            if oldLevel = 0 then
                                level.[vertex] <- level.[idx] + 1
                                frontOut.[vertex] <- true

                                edges.[i] <- true
                                goOn.[0] <- true

        /// <summary>
        /// Map edges to swips
        /// </summary>
        /// <param name="edges"> Edges of the partitioned graph spanning tree</param>
        /// <param name="len"> Number of edges </param>
        /// <param name="links"> Mapping of partitined edges -> real graph edges</param>
        /// <param name="swips"> Switching pairs of successors</param>
        [<Kernel; ReflectedDefinition>]
        let selectSwipsKernel (edges : deviceptr<bool>) len (links : deviceptr<int>) (swips : deviceptr<int>) =

            let idx = blockIdx.x * blockDim.x + threadIdx.x
            if idx < len && edges.[idx] then
                __atomic_exch (swips + links.[idx]) 1 |> ignore

        /// <summary>
        /// Generates spanning tree by bfs on the gpu
        //  In order to use weak connectivity, need to generate the undirected graph first
        /// </summary>
        /// <param name="gr"></param>
        let bfsGpu (gr : DirectedGraph<'a>) =

            let numEdges = gr.NumEdges
            let len = gr.NumVertices
            let lp = LaunchParam(divup len blockSize, blockSize)

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

            front.ScatterScalar true

            let getFront f = if f then front else frontOut

            let mutable flag = true

            while goOn.GatherScalar() do
                goOn.ScatterScalar(false)

                worker.Launch <@ bfsKernel @> lp (getFront flag).Ptr len (getFront (not flag)).Ptr dVisted.Ptr dLevel.Ptr dEdges.Ptr dRowIndex.Ptr dColIndex.Ptr goOn.Ptr
                flag <- not flag

            dEdges

        let bfs (gr : DirectedGraph<'a>) =
            bfsGpu gr |> fun dEdges -> dEdges.Gather()

        /// <summary>
        /// Generates swips from the partitioned graph and a mapping array
        /// From partitioned to real graph edges
        /// </summary>
        /// <param name="gr">Partitioned graph</param>
        /// <param name="links">Map from partitioned to original edges</param>
        /// <param name= "numOfOriginalGraphEdges">Number of original graph edges</param>
        let generateSwipsGpu (gr : DirectedGraph<'a>) (links : int []) numOfOriginalGraphEdges =
            let dSwips = worker.Malloc(Array.create numOfOriginalGraphEdges 0)
            use dLinks = worker.Malloc(links)
            let lp = LaunchParam(divup gr.NumEdges blockSize, blockSize)

            use dEdges = worker.Malloc(gr.SpanningTreeEdges)

            worker.Launch<@ selectSwipsKernel @> lp dEdges.Ptr gr.NumEdges dLinks.Ptr dSwips.Ptr
            dSwips

