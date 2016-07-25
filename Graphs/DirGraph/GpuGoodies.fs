namespace Graphs

module GpuGoodies =
    open Alea.CUDA
    open Alea.CUDA.Unbound
    open Alea.CUDA.Utilities

    let blockSize = 1024
    let worker = Worker.Default
    let target = GPUModuleTarget.Worker worker

    [<Kernel; ReflectedDefinition>]
    let copyGpu (source : deviceptr<'a>) (dest : deviceptr<'a>) len =
        let idx = blockIdx.x * blockDim.x + threadIdx.x
        if idx < len - 1 then
            dest.[idx] <- source.[idx]

    // represent the graph as two arrays. For each vertex v, an edge is a tuple
    // start[v], end'[v]
    [<Kernel;ReflectedDefinition>]
    let toEdgesKernel (rowIndex : deviceptr<int>) len (colIndex : deviceptr<int>) (start : deviceptr<int>) (end' : deviceptr<int>) =
        let idx = blockIdx.x * blockDim.x + threadIdx.x
        if idx < len - 1 then
            for vertex = rowIndex.[idx] to rowIndex.[idx + 1] - 1 do
                start.[vertex] <- idx
                end'.[vertex] <- colIndex.[vertex]

    [<Kernel; ReflectedDefinition>]
    let scatter (start : deviceptr<int>) (end' : deviceptr<int>) (len: int) (falsesScan : deviceptr<int>) (revBits : deviceptr<int>) (outStart : deviceptr<int>) (outEnd : deviceptr<int>) =
        let idx = blockIdx.x * blockDim.x + threadIdx.x
        if idx < len then

            let totalFalses = falsesScan.[len - 1] + revBits.[len - 1]

            // when the bit is equal to 1 - it will be offset by the scan value + totalFalses
            // if it's equal to 0 - just the scan value contains the right address
            let addr = if revBits.[idx] = 1 then falsesScan.[idx] else totalFalses + idx - falsesScan.[idx]
            outStart.[addr] <- start.[idx]
            outEnd.[addr] <- end'.[idx]

    [<Kernel; ReflectedDefinition>]
    let getNthSignificantReversedBit (arr : deviceptr<int>) (n : int) (len : int) (revBits : deviceptr<int>) =
        let idx = blockIdx.x * blockDim.x + threadIdx.x
        if idx < len then
            revBits.[idx] <- ((arr.[idx] >>> n &&& 1) ^^^ 1)

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


    let getEdgesGpu (rowIndex : int []) (colIndex : int []) =
        let len = rowIndex.Length

        let lp = LaunchParam(divup len blockSize, blockSize)

        let dStart = worker.Malloc<int>(colIndex.Length)
        let dEnd = worker.Malloc<int>(colIndex.Length)
        use dRowIndex = worker.Malloc(rowIndex)
        use dColIndex = worker.Malloc(colIndex)

        worker.Launch <@ toEdgesKernel @> lp dRowIndex.Ptr len dColIndex.Ptr dStart.Ptr dEnd.Ptr

        dStart, dEnd

    let getEdges (rowIndex : int []) (colIndex : int []) =
        let dStart, dEnd = getEdgesGpu rowIndex colIndex
        dStart.Gather(), dEnd.Gather()

    let getBitCount n =
        let rec getNextPowerOfTwoRec n acc =
            if n = 0 then acc
            else getNextPowerOfTwoRec (n >>> 1) (acc + 1)

        getNextPowerOfTwoRec n 0
    /// <summary>
    /// Sort by end vertices
    /// </summary>
    /// <param name="dStart"></param>
    /// <param name="dEnd"></param>
    let sortStartEndGpu (dStart : DeviceMemory<int>) (dEnd : DeviceMemory<int>) =
        let len = dStart.Length

        let lp = LaunchParam(divup len blockSize, blockSize)

        use reduceModule = new DeviceReduceModule<int>(target, <@ max @>)
        use reducer = reduceModule.Create(len)

        use scanModule = new DeviceScanModule<int>(target, <@ (+) @>)
        use scanner = scanModule.Create(len)

        use dBits = worker.Malloc(len)
        use numFalses = worker.Malloc(len)
        let dStartTemp = worker.Malloc(len)
        let dEndTemp = worker.Malloc(len)

        // Number of iterations = bit count of the maximum number
        let numIter = reducer.Reduce(dEnd.Ptr, len) |> getBitCount

        let getArr i = if i &&& 1 = 0 then dStart, dEnd else dStartTemp, dEndTemp
        let getOutArr i = getArr (i + 1)

        for i = 0 to numIter - 1 do
            // compute significant bits
            let start, end' = getArr i
            let outStart, outEnd = getOutArr i
            worker.Launch <@ getNthSignificantReversedBit @> lp end'.Ptr i len dBits.Ptr

            // scan the bits to compute starting positions further down
            scanner.ExclusiveScan(dBits.Ptr, numFalses.Ptr, 0, len)

            // scatter
            worker.Launch <@ scatter @> lp start.Ptr end'.Ptr len numFalses.Ptr dBits.Ptr outStart.Ptr outEnd.Ptr

        getOutArr (numIter - 1)

    let sortStartEnd (dStart : DeviceMemory<int>) (dEnd : DeviceMemory<int>) =
        let outStart, outEnd = sortStartEndGpu dStart dEnd
        outStart.Gather(), outEnd.Gather()

    let partitionGpu (dStart : DeviceMemory<int>) (dEnd: DeviceMemory<int>) nVertices =
        let lp = LaunchParam(divup dStart.Length blockSize, blockSize)

        let dColor = worker.Malloc([|0..nVertices - 1|])
        use dGo = worker.Malloc([|true|])

        while dGo.GatherScalar() do
            dGo.Scatter([|false|])
            worker.Launch <@ partionKernel @> lp dStart.Ptr dEnd.Ptr dColor.Ptr dStart.Length dGo.Ptr

        dColor