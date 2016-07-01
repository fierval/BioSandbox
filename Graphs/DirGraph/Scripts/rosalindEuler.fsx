#load "load-project-debug.fsx"

open Graphs
open System.IO

// Solving a Rosalind problem http://rosalind.info/problems/ba3f/
let rosgr = StrGraph.FromFile(@"c:\users\boris\downloads\rosalind_ba3f.txt")

let euler_path = rosgr.FindEulerPath()
let path_text = euler_path |> List.reduce (fun st e -> st + "->" + e)

File.WriteAllText(@"c:\temp\ros_ba3g.txt", path_text)

let gr3 = StrGraph.GenerateEulerGraph(3000, 10, path = true)
