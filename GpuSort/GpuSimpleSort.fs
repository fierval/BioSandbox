module GpuSimpleSort

open Alea.CUDA
open Alea.CUDA.Unbound
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound.Rng

open System.Diagnostics
open Microsoft.FSharp.Quotations
open System

let worker = Worker.Default
let target = GPUModuleTarget.Worker worker

let blockSize = 512

[<Kernel; ReflectedDefinition>]
let getNthSignificantReversedBit (arr : deviceptr<int>) (n : int) (len : int) (revBits : deviceptr<int>) =
    let idx = blockIdx.x * blockDim.x + threadIdx.x
    if idx < len then
        revBits.[idx] <- ((arr.[idx] >>> n &&& 1) ^^^ 1)

[<Kernel; ReflectedDefinition>]
let scatter (arr : deviceptr<int>) (len: int) (falsesScan : deviceptr<int>) (revBits : deviceptr<int>) (out : deviceptr<int>) =
    let idx = blockIdx.x * blockDim.x + threadIdx.x
    if idx < len then

        let totalFalses = falsesScan.[len - 1] + revBits.[len - 1]

        // when the bit is equal to 1 - it will be offset by the scan value + totalFalses
        // if it's equal to 0 - just the scan value contains the right address
        let addr = if revBits.[idx] = 1 then falsesScan.[idx] else totalFalses + idx - falsesScan.[idx]
        out.[addr] <- arr.[idx]

let getBitCount n =
    let rec getNextPowerOfTwoRec n acc =
        if n = 0 then acc
        else getNextPowerOfTwoRec (n >>> 1) (acc + 1)

    getNextPowerOfTwoRec n 0

let sortGpu (dArr: DeviceMemory<int>) =
    let len = dArr.Length
    if len = 0 then dArr
    else
        let gridSize = divup len blockSize
        let lp = LaunchParam(gridSize, blockSize)

        // reducer to find the maximum number & get the number of iterations
        // from it.
        use reduceModule = new DeviceReduceModule<int>(target, <@ max @>)
        use reducer = reduceModule.Create(len)

        use scanModule = new DeviceScanModule<int>(target, <@ (+) @>)
        use scanner = scanModule.Create(len)

        use dBits = worker.Malloc(len)
        use numFalses = worker.Malloc(len)
        let dArrTemp = worker.Malloc(len)

        // Number of iterations = bit count of the maximum number
        let numIter = reducer.Reduce(dArr.Ptr, len) |> getBitCount

        let getArr i = if i &&& 1 = 0 then dArr else dArrTemp
        let getOutArr i = getArr (i + 1)

        for i = 0 to numIter - 1 do
            // compute significant bits
            worker.Launch <@ getNthSignificantReversedBit @> lp (getArr i).Ptr i len dBits.Ptr

            // scan the bits to compute starting positions further down
            scanner.ExclusiveScan(dBits.Ptr, numFalses.Ptr, 0, len)

            // scatter
            worker.Launch <@ scatter @> lp (getArr i).Ptr len numFalses.Ptr dBits.Ptr (getOutArr i).Ptr

        getOutArr (numIter - 1)

let sort (arr : int []) =
    use dArr = worker.Malloc(arr)
    use sortedArr = sortGpu dArr
    sortedArr.Gather()


let generateRandomData n =
    if n <= 0 then failwith "n should be positive"
    let seed = uint32 DateTime.Now.Second

    // setup random number generator
    use cudaRandom = (new XorShift7.CUDA.DefaultNormalRandomModuleF32(target)).Create(1, 1, seed) :> IRandom<float32>
    use prngBuffer = cudaRandom.AllocCUDAStreamBuffer n

    // create random numbers
    cudaRandom.Fill(0, n, prngBuffer)
    // transfer results from device to host
    prngBuffer.Gather() |> Array.map (((*) (float32 n)) >> int >> (fun x -> if x = Int32.MinValue then Int32.MaxValue else abs x))

let toBin x =
    if x = 0 then "0" else
        0
        |> Seq.unfold (fun st -> if (x >>> st) = 0 then None else Some(x >>> st &&& 1, st + 1))
        |> Seq.toArray
        |> Array.rev
        |> Array.fold (fun st e -> st + string e) ""

let toBinRev x =
    if x = 0 then "0" else
        0
        |> Seq.unfold (fun st -> if (1 <<< st) > x then None else Some((x >>> st &&& 1) ^^^ 1, st + 1))
        |> Seq.toArray
        |> Array.rev
        |> Array.fold (fun st e -> st + string e) ""

let getBit x n =
    (x >>> n) &&& 0x1

