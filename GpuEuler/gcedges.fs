namespace GpuEuler

    [<AutoOpen>]
    module GCEdges =
        open Graphs
        open Alea.CUDA
        open Alea.CUDA.Utilities
        open Graphs.GpuGoodies
        open System.Collections.Generic
        open GpuCompact

        [<Literal>]
        let MaxColros = 1000

        /// <summary>
        /// Takes the graph rowIndex and generates an array of valid swaps
        /// </summary>
        /// <param name="rowIndex">rowIndex of a reverse eulerian graph</param>
        /// <param name="colors">partition created in the previous step</param>
        let generateCircuitGraphLinear (rowIndex : int []) (colors : int []) =
            let numColors = 1 + (colors |> Array.max)
            let numVertices = rowIndex.Length - 1
            let numEdges = rowIndex.[rowIndex.Length - 1]

            let status = Array.create numColors false
            let validity = Array.create numEdges true
            let ea = List<int>()
            let eb = List<int>()
            let links = List<int>()
            let mutable pre = 0

            for i = 0 to numVertices - 1 do
                for j = rowIndex.[i] to rowIndex.[i + 1] - 1 do
                    if not status.[colors.[j]] then
                        status.[colors.[j]] <- true
                    else
                        validity.[j] <- false

                pre <- rowIndex.[i]
                for j = rowIndex.[i] + 1 to rowIndex.[i + 1] - 1 do
                    if validity.[j] then
                        ea.Add colors.[j]
                        eb.Add colors.[pre]
                        links.Add pre
                        pre <- j

                Array.fill status 0 status.Length false

            // convert to graph
            let eaArr = ea.ToArray()
            let ebArr = eb.ToArray()

            ea.AddRange ebArr
            eb.AddRange eaArr
            links.AddRange links

            ea.ToArray(), eb.ToArray(), links.ToArray(), validity

        [<Kernel; ReflectedDefinition>]
        let circuitGraphKernel (rowIndex : deviceptr<int>) numVertices (links : deviceptr<int>) (validity : deviceptr<int>) (ea : deviceptr<int>) (eb : deviceptr<int>) (colors : deviceptr<int>) =

            let idx = blockDim.x * blockIdx.x + threadIdx.x
            if idx < numVertices then
                let status : bool [] = __local__.Array(MaxColros)
                for i = 0 to status.Length do
                    status.[i] <- false

                let mutable pre = 0

                for j = rowIndex.[idx] to rowIndex.[idx + 1] - 1 do
                    if not status.[colors.[j]] then
                        status.[colors.[j]] <- true
                    else
                        validity.[j] <- 0

                pre <- rowIndex.[idx]
                for j = rowIndex.[idx] + 1 to rowIndex.[idx + 1] - 1 do
                    if validity.[j] = 1 then
                        ea.[j] <- colors.[j]
                        eb.[j] <- colors.[pre]
                        links.[j] <- pre
                        pre <- j

        let generateCircuitGraphGpu (rowIndex : int []) (colors : int []) =
            let numVertices = rowIndex.Length - 1
            let lp = LaunchParam(divup numVertices blockSize, blockSize)
            let numEdges = rowIndex.[rowIndex.Length - 1]

            let zeros : int [] = Array.zeroCreate numEdges
            let ones : int [] = Array.create numEdges 1

            use dLinks = worker.Malloc<int>(zeros)
            use dValidity = worker.Malloc<int>(ones)
            use dEa = worker.Malloc<int>(zeros)
            use dEb = worker.Malloc<int>(zeros)
            use dColors = worker.Malloc(colors)
            use dRowIndex = worker.Malloc(rowIndex)

            worker.Launch <@circuitGraphKernel@> lp dRowIndex.Ptr numVertices dLinks.Ptr dValidity.Ptr dEa.Ptr dEb.Ptr dColors.Ptr

            let compact (a : DeviceMemory<int>) = a |> compactGpu |> (fun c -> c.Gather())

            let ea = compact dEa
            let eb = compact dEb
            let links = compact dLinks
            let validityInt = compact dValidity

            ea, eb, links, (validityInt |> Array.map (fun e -> e = 1))

        /// <summary>
        /// Takes the graph rowIndex and generates an array of valid swaps
        /// </summary>
        /// <param name="rowIndex">rowIndex of a reverse eulerian graph</param>
        /// <param name="colors">partition created in the previous step</param>
        let generateCircuitGraph (rowIndex : int[]) (colors : int []) =
            let ea, eb, links, validity =
                if colors.Length < 1000 then
                    generateCircuitGraphGpu rowIndex colors
                else
                    generateCircuitGraphLinear rowIndex colors

            let starts, colIndex, linkPos =
                Seq.zip3 ea eb links
                |> Seq.sortBy (fun (a, _, _) -> a)
                |> Seq.toArray
                |> Array.unzip3

            let newRowIndex =
                starts
                |> Array.groupBy id
                |> Array.map (fun (_, v) -> v.Length)
                |> Array.scan (+) 0

            (StrGraph.FromInts newRowIndex colIndex), linkPos, validity
