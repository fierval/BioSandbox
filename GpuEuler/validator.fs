namespace GpuEuler

[<AutoOpen>]
module Validator =
    open Graphs
    open System.Diagnostics
    open System.Linq

    let validate (gr : DirectedGraph<'a>) (edges : int []) =
        let grEdges = gr.OrdinalEdges

        // every edge is traversed exactly once
        edges.Length = gr.NumEdges
        && edges.Distinct().ToArray().Length = gr.NumEdges
        && (
            // there is actually an eulerian cycle
            let eulerGraph = StrGraph.FromVectorOfInts(edges).Reverse

            eulerGraph.OrdinalEdges
            |> Array.map (fun (s, e) -> snd grEdges.[s] = fst grEdges.[e])
            |> Array.exists not
            |> not
        )




