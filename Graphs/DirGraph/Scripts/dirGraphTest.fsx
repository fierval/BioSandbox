#load "load-project-debug.fsx"

open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics
open System.IO
open DrawGraph

let strs = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"]
let strs1 = ["a -> c, d"; "b -> a, c"; "d -> e, f"; "e -> f"]

let gr = StrGraph.FromStrings strs
let gr1 = StrGraph.FromStrings strs1
let gr2 = StrGraph.FromStrings strs


printfn "%b" (gr = gr1)
printfn "%b" (gr = gr2)

let sparse = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"; "1 -> 2, 3"; "3 -> 4, 5"; "x -> y, z"; "2 -> 5"]
let grs = StrGraph.FromStrings sparse

//let rosgr = StrGraph.FromFile(@"C:\Users\boris\Downloads\eulerian_cycle.txt")

let euler = StrGraph.GenerateEulerGraph(12, 3, path=true)

//grs.Visualize()
//grs.Reverse.Visualize()
grs.Visualize(clusters = true)
//StrGraph.Visualize(gr, clusters = true)
euler.Visualize(euler=true)

let strsr = ["0 -> 3"; "1 -> 0"; "2 -> 1,6"; "3 -> 2"; "4 -> 2"; "5 -> 4"; "6 -> 5,8"; "7 -> 9"; "8 -> 7"; "9 -> 6"]
let grr = StrGraph.FromStrings strsr

//grr.Visualize(euler=true)
grs.Edges
grs.Partition()