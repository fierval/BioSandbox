#load "Scripts/load-project-debug.fsx"

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