#load "load-project-release.fsx"

open Alea.CUDA
open System.IO
open System
open System.Collections.Generic
open System.Linq

open GpuCompact
open GpuDistinct
open System.Diagnostics

let mutable N = 100
Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let rnd = Random()
let sw = Stopwatch()

// warm up the GPU
let mutable sample = Array.init N (fun i -> rnd.Next(0, 1000))
GpuDistinct.distinct sample

for i = 1 to 3 do
    N <- (pown 10 i) / 4  * 1024 * 1024
    printfn "%s" (String.Format("Length: {0:N0}", N))

    sample <- Array.init N (fun i -> rnd.Next(0, 1000))

    sw.Restart()
    sample |> Array.distinct |> ignore
    sw.Stop()
    printfn "CPU distinct: %A" sw.Elapsed

    sw.Restart()
    GpuDistinct.distinct sample |> ignore
    sw.Stop()
    printfn "GPU distinct: %A" sw.Elapsed

N <-  300000000
sample <- Array.init N (fun i -> rnd.Next(0, 1000))

sw.Restart()
sample |> Array.distinct |> ignore
sw.Stop()
printfn "CPU distinct: %A" sw.Elapsed

sw.Restart()
GpuDistinct.distinct sample |> ignore
sw.Stop()
printfn "GPU distinct: %A" sw.Elapsed
