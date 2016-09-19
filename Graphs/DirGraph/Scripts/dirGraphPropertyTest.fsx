#load "load-project-release.fsx"

open Graphs
open FsCheck
open System
open System.Text.RegularExpressions
open System.Diagnostics
open DataGen

type Marker =
    static member digr = graphGen 3 500 |> Arb.fromGen
    static member ``Reverse of the Reverse equals itself`` (gr : DirectedGraph<string>) =
        gr.Reverse.Reverse = gr

Arb.registerByType(typeof<Marker>)
Check.QuickAll(typeof<Marker>)

let grGen = graphGen 3 50

let gr = grGen.Sample(15, 5).[2]
gr.Visualize(into=3, out= 3)

let gre = StrGraph.GenerateEulerGraph(10, 5)
gre.Visualize(into=3, out=3)
