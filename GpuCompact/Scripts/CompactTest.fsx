#load "load-project-debug.fsx"

open Alea.CUDA
open System.IO
open System
open System.Collections.Generic
open System.Linq

open GpuCompact
open GpuDistinct

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\packages\Alea.Cuda.2.2.0.3307\private")
Alea.CUDA.Settings.Instance.Resource.Path <- Path.Combine(__SOURCE_DIRECTORY__, @"..\..\release")

let rnd = Random(int DateTime.Now.Ticks)
let nums = List<int>()
let range = [0..rnd.Next(100, 10000)]

for i in range do
    nums.Add (rnd.Next(0, rnd.Next(1, 1000)))

let arr = nums.ToArray()

let dist = arr |> Array.distinct |> Array.sort

let dist1 = distinct arr

dist = dist1

