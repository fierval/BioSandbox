#load "load-project-debug.fsx"

open Graphs
open GpuEuler

let gr = StrGraph.GenerateEulerGraphAlt(15, 36)
gr.Visualize(euler=true)

