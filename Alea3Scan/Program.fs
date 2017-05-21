open Alea
open Alea.Parallel
open System

[<EntryPoint>]
let main argv = 
    
    let len = 100 * 100
    let arr = [|0..len - 1|]
    let mutable arrOut = Array.zeroCreate len
    let gpu = Gpu.Default;
    use session = new Session(gpu)
    use devIn = gpu.AllocateDevice(arr)
    use devOut = gpu.AllocateDevice(arrOut)
    
    session.Scan(devOut.Ptr, devIn.Ptr, 0, len, (fun x y -> x + y), 0)
    arrOut <- Gpu.CopyToHost(devOut)
    printfn "%A" arrOut
    
    let arrScan = arr |> Array.scan(fun st y -> st + y) 0
    printfn "%b" (arrOut = arrScan.[1..])

    0 // return an integer exit code

