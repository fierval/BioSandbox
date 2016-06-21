module BlockScanSample

open Alea.CUDA
open Alea.CUDA.Unbound
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound.Rng

open System.Diagnostics
open Microsoft.FSharp.Quotations
open System

open GpuSimpleSort
let arch = worker.Device.Arch

type ScanPrimitive(arch:DeviceArch, op:Expr<'T -> 'T -> 'T>) =

    let blockRangeScan = DeviceScanPolicy.Create(arch, PlatformUtil.Instance.ProcessBitness, None).BlockRangeScan

    [<ReflectedDefinition>]
    member this.BlockRangeScan blockOffset blockEnd (inputs:deviceptr<_>) (outputs : deviceptr<_>) =
        let tempStorage = blockRangeScan.TempStorage.AllocateShared()
        blockRangeScan.ConsumeRangeConsecutiveExclusive
            tempStorage (Iterator inputs) (Iterator outputs) (__eval op) 0 blockOffset blockEnd

    member this.BlockThreads = blockRangeScan.BlockThreads

type BlockScanModule(target, op:Expr<'T -> 'T -> 'T>) as this =
    inherit GPUModule(target)

    let primitive = 
        fun (options:CompileOptions) ->
            cuda { return ScanPrimitive(arch, op) }
        |> this.GPUDefineResource

    [<Kernel;ReflectedDefinition>]
    member this.Kernel blockSize (inputs:deviceptr<_>) (outputs:deviceptr<_>) =
        let blockOffset = blockIdx.x * blockSize
        primitive.Resource.BlockRangeScan blockOffset (blockOffset + blockSize) inputs outputs

    member this.Apply(arr : 'T []) =
        printfn "# threads: %d" primitive.Resource.BlockThreads

        let dInp = worker.Malloc(arr)
        let dOut = worker.Malloc<'T>(arr.Length)
        let lp = LaunchParam(divup arr.Length primitive.Resource.BlockThreads, primitive.Resource.BlockThreads)
        this.GPULaunch <@ this.Kernel @> lp primitive.Resource.BlockThreads dInp.Ptr dOut.Ptr
        dOut.Gather()

let bs = new BlockScanModule(target, <@ (+) @>)

//let sortBitonic (arr : int []) =
//    let len = arr.Length
//    if len = 0 then [||]
//    else
//        let gridSize = divup len blockSize
//        let lp = LaunchParam (gridSize, blockSize)
//
//        // reducer to find the maximum number & get the number of iterations
//        // from it.
//        use reduceModule = new DeviceReduceModule<int>(target, <@ max @>)
//        use reducer = reduceModule.Create(len)
//
//        BlockScan<int>(dim3(blockSize),)
//        use scanModule = new GPUModule(target)
//        use scanner = scanModule.Create(len)
//
//        use dArr = worker.Malloc(arr)
//        use dBits = worker.Malloc(len)
//        use numFalses = worker.Malloc(len)
//        use dArrTemp = worker.Malloc(len)
//
//        // Number of iterations = bit count of the maximum number
//        let numIter = reducer.Reduce(dArr.Ptr, len) |> getBitCount
//
//        let getArr i = if i &&& 1 = 0 then dArr else dArrTemp
//        let getOutArr i = getArr (i + 1)
//
