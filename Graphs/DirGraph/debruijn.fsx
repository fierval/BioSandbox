#r @"..\..\packages\FsCheck.2.4.0\lib\net45\FsCheck.dll"
#r @"..\..\packages\Alea.CUDA.2.2.0.3307\lib\net40\Alea.CUDA.dll"
#r @"..\..\packages\Alea.CUDA.Unbound.2.2.0.3307\lib\net40\Alea.CUDA.Unbound.dll"
#r @"C:\Git\BioSandbox\Graphs\DrawGraph\bin\Debug\DrawGraph.dll"
#I @"..\..\packages\Alea.CUDA.2.2.0.3307\lib\net40"
#I @"..\..\packages\Alea.CUDA.Unbound.2.2.0.3307\lib\net40"

#r @"Alea.CUDA"
#r @"Alea.CUDA.Unbound"

#nowarn "25"

#load "dirGraph.fs"
#load "visualizer.fs"

open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics
open DirectedGraph

let prefix (s:string) = s.[0..s.Length - 2]
let suffix (s:string) = s.[1..]
let numToBinary n len =
    let rec numToBinaryRec n len acc =
        if len = 0 then acc
        else
           numToBinaryRec (n >>> 1) (len - 1) (String.Format("{0}{1}", n &&& 0x1, acc))
    numToBinaryRec n len ""

let binaryDebruijnSeq n =
    if n <= 0 then failwith "n should be positive"
    let finish = int (2.0 ** (float n))
    let graphStrs =
        [0..finish-1] 
        |> List.map (fun i -> numToBinary i n)
        |> List.map (fun s -> prefix s, suffix s)
        |> List.groupBy fst
        |> List.map (fun (v, prefSuf) -> v + " -> " + (prefSuf |> List.map snd |> List.reduce (fun st e -> st + "," + e )))
        |> DirectedGraph<string>.FromStrings
    
    let debruinSeq = graphStrs.FindEulerPath()
    let debruinNum = debruinSeq |> List.windowed 2 |> List.mapi (fun i [p; s] -> "\"" + (i + 1).ToString() + ":" + s.[s.Length - 1].ToString() + "\"")

    Visualizer.Visualize(graphStrs, euler = true, eulerLabels = debruinNum)
