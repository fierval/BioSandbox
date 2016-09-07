#load "load-project-debug.fsx"

open Graphs
open GpuEuler

let gr = StrGraph.GenerateEulerGraphAlt(2, 10)
gr.Visualize(euler=true)

