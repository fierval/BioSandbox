open Alea
open Alea.Parallel
open System

[<EntryPoint; GpuManaged>]
let main argv = 
    
    let len = 100 * 100
    let arr = [|0..len - 1|]
    let arrOut = Array.zeroCreate len
    let gpu = Gpu.Default;
    use session = new Session(gpu)

    session.Scan(arrOut, arr, 0, (fun x y -> x + y), 0)
    printfn "%A" arrOut

    0 // return an integer exit code

