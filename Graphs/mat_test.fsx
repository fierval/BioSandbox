// Learn more about F# at http://fsharp.org. See the 'F# Tutorial' project
// for more guidance on F# programming.
#r @"..\packages\Alea.CUDA.2.2.0.3307\lib\net40\Alea.CUDA.dll"
#r @"..\packages\Alea.CUDA.IL.2.2.0.3307\lib\net40\Alea.CUDA.IL.dll"
#r @"..\packages\Alea.CUDA.Unbound.2.2.0.3307\lib\net40\Alea.CUDA.Unbound.dll"
#r @"..\packages\Alea.IL.2.2.0.3307\lib\net40\Alea.IL.dll"
#r @"C:\Program Files (x86)\FSharpPowerPack-4.0.0.0\bin\FSharp.PowerPack.dll"

// Testing
#r @"..\packages\NUnit.3.2.1\lib\net45\nunit.framework.dll"
#r @"..\packages\FsCheck.2.4.0\lib\net45\FsCheck.dll"

#load "SparseMatrix.fs"
open Graphs

// Testing
open FsCheck
open System
open NUnit.Framework


// Define your library scripting code here

let m = SparseMatrix.CreateMatrix [|1; 0; 0; 0; 0; 3; 4; 0; 0; 0; 1; 1|] true;;

m.AddValues([|0; 0; 2; 0; 1; 0; 5; 0; 3; 0; 2; 0|])
m.AddValues([|0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 4; 0|])

m.NNZ

printfn "%A" m