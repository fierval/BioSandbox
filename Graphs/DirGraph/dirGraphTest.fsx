// Learn more about F# at http://fsharp.org. See the 'F# Tutorial' project
// for more guidance on F# programming.

#r @"..\..\packages\FsCheck.2.4.0\lib\net45\FsCheck.dll"
#r @"C:\Git\BioSandbox\Graphs\DrawGraph\bin\Debug\DrawGraph.dll"

#load "dirGraph.fs"
open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics


//let nucl = Gen.oneof [gen {return 'A'}; gen {return 'C'}; gen {return 'G'}; gen {return 'T'}]
let nucl = Gen.choose(int 'A', int 'Z') |> Gen.map char

let genVertex len =  Gen.arrayOfLength len nucl |> Gen.map (fun c -> String(c))
let vertices len number = Gen.arrayOfLength number (genVertex len) |> Gen.map (fun l -> l |> Array.distinct)

let connections len number =
    let verts = vertices len number
    let rnd = Random(int DateTime.UtcNow.Ticks)
    let pickFrom = verts |> Gen.map (fun lst -> lst.[rnd.Next(lst.Length)])
    let pickTo = Gen.sized (fun n -> Gen.listOfLength (if n = 0 then 1 else n) pickFrom)

    Gen.map2 
        (fun from to' -> 
            from,
                (to' |> Seq.reduce (fun acc v -> acc + ", " + v))) pickFrom pickTo

let graphGen = connections 4 1000
let sw = Stopwatch()
sw.Start()
let graph = graphGen.Sample(5, 1000) |> List.distinctBy fst |> List.map (fun (v, c) -> v + " -> " + c)
sw.Stop()

printfn "Took %A to generate a graph of %d vertices" sw.Elapsed graph.Length
open Graphs

let digr = DirectedGraph.FromStrings graph
digr.Visualize(emphasizeOutConnections = 5);;
