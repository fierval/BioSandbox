#load "Scripts/load-project-debug.fsx"

open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics
open System.IO
open DrawGraph

let strs = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"]
let strs1 = ["a -> c, d"; "b -> a, c"; "d -> e, f"; "e -> f"]

type StrGraph = DirectedGraph<string>
let gr = StrGraph.FromStrings strs
let gr1 = StrGraph.FromStrings strs1
let gr2 = StrGraph.FromStrings strs


printfn "%b" (gr = gr1)
printfn "%b" (gr = gr2)

let sparse = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"; "1 -> 2, 3"; "3 -> 4, 5"; "x -> y, z"; "2 -> 5"]
let grs = StrGraph.FromStrings sparse

//let rosgr = StrGraph.FromFile(@"C:\Users\boris\Downloads\eulerian_cycle.txt")

let euler = StrGraph.GenerateEulerGraph(12, 3, path=true)

Visualizer.Visualize(grs)
Visualizer.Visualize(grs.Reverse)
Visualizer.Visualize(grs, clusters = true)
Visualizer.Visualize(gr, clusters = true)
Visualizer.Visualize(euler, euler=true)

let strsr = ["0 -> 3"; "1 -> 0"; "2 -> 1,6"; "3 -> 2"; "4 -> 2"; "5 -> 4"; "6 -> 5,8"; "7 -> 9"; "8 -> 7"; "9 -> 6"]
let grr = StrGraph.FromStrings strsr

Visualizer.Visualize(grr, euler=true)

// Solving a Rosalind problem http://rosalind.info/problems/ba3f/
let rosgr = StrGraph.FromFile(@"c:\users\boris\downloads\rosalind_ba3f.txt")

let euler_path = rosgr.FindEulerPath()
let path_text = euler_path |> List.reduce (fun st e -> st + "->" + e)

File.WriteAllText(@"c:\temp\ros_ba3g.txt", path_text)

let gr3 = StrGraph.GenerateEulerGraph(3000, 10, path = true)
