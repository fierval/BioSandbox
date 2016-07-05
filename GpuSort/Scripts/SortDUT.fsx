#load @"load-project-debug.fsx"

open FsCheck
open GpuSimpleSort
open System.IO
open Alea.CUDA

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let genNonNeg = Arb.generate<int> |> Gen.filter ((<=) 0)

type ``Sort DUT`` =
    static member arbNonNeg = genNonNeg |> Arb.fromGen
    static member ``Sorting Correctly`` arr =
        sort arr = Array.sort arr

Arb.registerByType(typeof<``Sort DUT``>)
Check.QuickAll(typeof<``Sort DUT``>)

