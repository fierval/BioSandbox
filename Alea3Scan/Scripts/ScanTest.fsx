#r @"..\..\packages\Alea.3.0.3\lib\net45\Alea.dll"
#r @"..\..\packages\Alea.3.0.3\lib\net45\Alea.IL.dll"
#r @"..\..\packages\Alea.3.0.3\lib\net45\Alea.Parallel.dll"
#r @"System.Configuration"

open Alea
open Alea.Parallel
open System.IO

Alea.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\bin\debug")

let len = 100 * 100
let arr = [|0..len - 1|]
let mutable arrOut : int [] = Array.zeroCreate len
let gpu = Gpu.Default;
let session = new Session(gpu)
let devIn = gpu.AllocateDevice(arr)
let devOut = gpu.AllocateDevice(arrOut)

session.Scan(devOut.Ptr, devIn.Ptr, 0, len, (fun x y -> x + y), 0)
arrOut <- Gpu.CopyToHost(devOut)
printfn "%A" arrOut
    
let arrScan = arr |> Array.scan(fun st y -> st + y) 0
printfn "%b" (arrOut = arrScan.[1..])
