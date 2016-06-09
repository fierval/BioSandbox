#load @"Scripts\load-project-debug.fsx"

open FsCheck
open GpuSort

let genNonNeg = Arb.generate<int> |> Gen.filter ((<=) 0)

type Marker =
    static member arbNonNeg = genNonNeg |> Arb.fromGen
    static member ``Sorting Correctly`` arr =
        let carr = sort arr
        carr = Array.sort arr

Arb.registerByType(typeof<Marker>)
Check.QuickAll(typeof<Marker>)

