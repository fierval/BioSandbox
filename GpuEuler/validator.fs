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


    /// <summary>
    /// Convert the euler path represented as numbered edges to vertices path
    /// </summary>
    /// <param name="gr">Original graph</param>
    /// <param name="edges">Euler cycle as numbered edges</param>
    let toVertexPath (gr : DirectedGraph<'a>) (edges : int []) =
        let grEdges = gr.Edges

        let mutable j = 0
        let mutable vertices = [fst grEdges.[j]; snd grEdges.[j]]
        for i = 2 to gr.NumEdges do
            j <- edges.[j]
            vertices <- fst grEdges.[j]::vertices
        vertices

