#load @"Scripts\load-project-debug.fsx"

open FsCheck
open BlockScanSample
open System.IO
open Alea.CUDA


Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\release")

let genNonNeg = Arb.generate<int> |> Gen.filter ((<=) 0)
