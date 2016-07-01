#load "load-project-debug.fsx"

#nowarn "25"

open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics

let prefix (s:string) = s.[..s.Length - 2]
let suffix (s:string) = s.[1..]
let prefSuf s = prefix s, suffix s // shorthand

let numToBinary len n =
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
        |> List.map (numToBinary n >> prefSuf)
        |> List.groupBy fst
        |> List.map (fun (v, prefSuf) -> v + " -> " + (prefSuf |> List.map snd |> List.reduce (fun st e -> st + "," + e )))
        |> DirectedGraph<string>.FromStrings
    
    let debruinSeq = graphStrs.FindEulerPath()
    let debruinNum = debruinSeq |> List.windowed 2 |> List.mapi (fun i [p; s] -> "\"" + (i + 1).ToString() + ":" + s.[s.Length - 1].ToString() + "\"")

    graphStrs.Visualize(euler = true, eulerLabels = debruinNum)
