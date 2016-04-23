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

let genZero = gen {return 0}
let zeroOrNot = Gen.frequency [(3, genZero); (1, Gen.choose(1, 10))]

let generateRow  len = Gen.arrayOfLength len zeroOrNot
let generateZeroRow len = Gen.arrayOfLength len genZero
let genSingleRow len = Gen.frequency [(7, generateRow len); (1, generateZeroRow len)]

type Generators =
    static member SparseMatrix () =
        let len = 10
        let csr = true
        let matrixCreate = Gen.map (fun row -> SparseMatrix.CreateMatrix row csr) (genSingleRow len)

        let rec genMatrix mat size =
            match size with
            | 1 -> mat
            | s when s > 1 -> 
                let mat = 
                    Gen.map2 
                        (fun (m : SparseMatrix<int>) row -> 
                            m.AddValues row
                            m) mat (genSingleRow len)
                genMatrix mat (s - 1)
            | _ -> failwith "0 or negatives not allowed"
        
        fun s -> 
            let sz = if s = 0 then 1 else s 
            Gen.resize sz (genMatrix matrixCreate sz)
        |> Gen.sized 
        |> Arb.fromGen

Arb.register<Generators>()

// Define your library scripting code here
//let mR = SparseMatrix.CreateMatrix sampleRows.[0] true

