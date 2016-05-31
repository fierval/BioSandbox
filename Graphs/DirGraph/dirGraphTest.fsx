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
open DirectedGraph

let strs = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"]
let strs1 = ["a -> c, d"; "b -> a, c"; "d -> e, f"; "e -> f"]

let gr = DirectedGraph.FromStrings strs
let gr1 = DirectedGraph.FromStrings strs1

printfn "%b" (gr = gr1)

let dirg = DirectedGraph.GenerateEulerGraph 100 5
let sparse = ["a -> b, c, d"; "b -> a, c"; "d -> e, f"; "e -> f"; "1 -> 2, 3"; "3 -> 4, 5"; "x -> y, z"; "2 -> 5"]
let grs = DirectedGraph.FromStrings sparse

//let rosgr = DirectedGraph.FromFile(@"C:\Users\boris\Downloads\eulerian_cycle.txt")
//let euler = DirectedGraph.GenerateEulerGraph 100 3

let euler = DirectedGraph.GenerateEulerGraph 8 3

Visualizer.Visualize(grs)
Visualizer.Visualize(grs.Reverse)
Visualizer.Visualize(dirg, into = 5, out = 5)
Visualizer.Visualize(grs, clusters = true)
Visualizer.Visualize(gr, clusters = true)
Visualizer.Visualize(euler, euler=true)
