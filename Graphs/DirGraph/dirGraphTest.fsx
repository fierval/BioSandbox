#r @"..\..\packages\FsCheck.2.4.0\lib\net45\FsCheck.dll"
#r @"..\..\packages\Alea.CUDA.2.2.0.3307\lib\net40\Alea.CUDA.dll"
#r @"..\..\packages\Alea.CUDA.Unbound.2.2.0.3307\lib\net40\Alea.CUDA.Unbound.dll"
#r @"C:\Git\BioSandbox\Graphs\DrawGraph\bin\Debug\DrawGraph.dll"
#I @"..\..\packages\Alea.CUDA.2.2.0.3307\lib\net40"
#I @"..\..\packages\Alea.CUDA.Unbound.2.2.0.3307\lib\net40"

#r @"Alea.CUDA"
#r @"Alea.CUDA.Unbound"

#load "dirGraph.fs"
#load "visualizer.fs"

open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics

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

let euler = StrGraph.GenerateEulerGraph(10, 3, path=true)

Visualizer.Visualize(grs)
Visualizer.Visualize(grs.Reverse)
Visualizer.Visualize(grs, clusters = true)
Visualizer.Visualize(gr, clusters = true)
Visualizer.Visualize(euler, euler=true)
