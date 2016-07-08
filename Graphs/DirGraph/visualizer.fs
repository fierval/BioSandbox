﻿namespace Graphs

open System.Collections.Generic
open System.Linq
open System.IO
open System
open DrawGraph
open System.Linq
open Alea.CUDA
#nowarn "25"

[<AutoOpen>]
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Visualizer =

    type GraphSeq<'a> = seq<'a * 'a []>
    let internal nDotThreshold = 10
    let mutable internal whiteWashNonSpanningEdges = true

    let internal toColor (c : string) (vertices : seq<'a>) =
        if vertices |> Seq.isEmpty then "" else
        let formatstr =
            let f = "{0} [style=filled, color={1}"
            if c = "blue" then f + ", fontcolor=white]" else f + "]"
        vertices
        |> Seq.map (fun v -> String.Format(formatstr, v, c))
        |> Seq.reduce (+)

    let internal coloring (self: GraphSeq<'a>) (selfRev : GraphSeq<'a>) in' out =
        let bottomOutgoing (graphSeq : GraphSeq<'a>) bottom =
            if bottom = 0 then Seq.empty
            else
                graphSeq
                |> Seq.filter (fun (_, con) -> con.Length >= bottom)
                |> Seq.map fst

        let outVertices = bottomOutgoing self out
        let inVertices = bottomOutgoing selfRev in'
        let outInVertices = inVertices.Intersect outVertices

        outVertices.Except outInVertices |> toColor "green",
        inVertices.Except outInVertices |> toColor "yellow",
        outInVertices |> toColor "blue"

    // Create a "graph" or a "cluster" visualization based on clusterN parameter
    let internal visualizeSubgraph (subgraph : GraphSeq<'a>) (subgraphRev : GraphSeq<'a>) outConMin inConMin clusterN (spanEdges : HashSet<'a * 'a>) =
        let graphOpen = if clusterN >= 0 then "subgraph cluster_" + clusterN.ToString() + "{ color = red; " else "digraph {"
        let graphClose = "}"

        let colorOut, colorIn, colorBoth = coloring subgraph subgraphRev inConMin outConMin

        let edges = spanEdges.ToList()

        // used when we have a spanning tree to display
        let genEdge (out : 'a) (in' : 'a) =
            if spanEdges.Count = 0  then
                String.Format("{0} -> {1}", out, in')
            elif not (spanEdges.Contains((out, in'))) then
                String.Format("{0} -> {1} [color={2}]", out, in', if whiteWashNonSpanningEdges then "transparent" else "blue")
            else
                String.Format("{0} -> {1} [color = red]", out, in')

        let visualizable =
            subgraph
            |> Seq.map
                (fun (v, c) ->
                    if c.Length = 0 then v.ToString()
                    else
                        let out = v.ToString()
                        c
                        |> Array.map (fun in' ->
                            genEdge v in'
                            )

                        |> Array.reduce (fun acc e -> acc + "; " + e)
                        )
            |> Seq.reduce (fun acc e -> acc + "; " + e)
            |> fun v -> graphOpen + colorOut + colorIn + colorBoth + v + graphClose

        visualizable

    let internal visualizeEntire subgraph subgraphRev outConMin inConMin spanning visualizer =
        visualizeSubgraph subgraph subgraphRev outConMin inConMin -1 spanning
        |> visualizer

    let internal visualizeAll (graph : DirectedGraph<'a>) outConMin inConMin clusters euler eulerLabels spanning=
        if euler then
            let eulerPath = graph.FindEulerPath()
            if eulerPath |> Seq.isEmpty then failwith "Graph not Eulerian"

            let color i = if i &&& 0x1 = 0 then "green" else "red"
            let labels = eulerLabels |> Seq.toList
            let label i = if not (List.isEmpty labels) then labels.[i] else (i + 1).ToString()

            eulerPath
            |> Seq.windowed 2
            |> Seq.mapi (fun i [|out; in'|] -> String.Format("{0} -> {1} [label = {2}, fontcolor = blue, color={3}]", out, in', label i, color i))
            |> Seq.reduce (fun acc e -> acc + "; " + e)
            |> fun gr ->
                eulerPath.[0].ToString() + "[color=green, style=filled]; " +
                    (if eulerPath.[0] <> eulerPath.[eulerPath.Length - 1] then
                        eulerPath.[eulerPath.Length - 1].ToString() + "[color=red, style=filled]; " else String.Empty) + gr
            |> fun gr -> ("digraph { " + gr + "}") |> if graph.NumVertices <= nDotThreshold then visualizeDot else visualizeSfdp

        else
            let rev = graph.Reverse
            let self = graph.AsEnumerable
            let selfRev = rev.AsEnumerable

            let connectedComponents =
                if clusters then
                    graph.FindConnectedComponents()
                    |> Array.map (fun h -> h.AsEnumerable() |> Seq.toList)
                else [||]

            let spanEdges = if spanning then graph.SpanningTree else HashSet<'a * 'a>()

            if not clusters then visualizeEntire self selfRev outConMin inConMin spanEdges (if graph.NumVertices <= nDotThreshold then visualizeDot else visualizeSfdp)
            else
                connectedComponents
                |> Array.mapi
                    (fun i vertices ->
                        let subgraph = vertices |> graph.Subgraph
                        let subgraphRev = vertices |> rev.Subgraph
                        visualizeSubgraph subgraph subgraphRev outConMin inConMin i (HashSet<'a * 'a>())
                    )
                |> Array.reduce (+)
                |> fun gr -> createVisualClusters ("digraph { " + gr + "}")

    /// <summary>
    /// Visualizer extenstion
    /// </summary>
    type DirectedGraph<'a when 'a: comparison> with

        /// <summary>
        /// Visualize the graph. Should in/out connections be emphasized
        /// </summary>
        /// <param name="into">Optional. If present - should be the minimum number of inbound connections which would select the vertex for coloring.</param>
        /// <param name="out">Optional. If present - should be the minimum number of outbound connections which would select the vertex for coloring.</param>
        member graph.Visualize(?into, ?out, ?clusters, ?euler, ?eulerLabels : string seq, ?spanningTree : bool) =
            let outConMin = defaultArg out 0
            let inConMin = defaultArg into 0
            let clusters = defaultArg clusters false
            let euler = defaultArg euler false
            let eulerLabels = defaultArg eulerLabels Seq.empty
            let spanning = defaultArg spanningTree false
            whiteWashNonSpanningEdges <- graph.CleanSpanningTree

            visualizeAll graph outConMin inConMin clusters euler eulerLabels spanning
