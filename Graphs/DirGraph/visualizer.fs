namespace Graphs

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
    
    type GraphSeq = seq<string * string []>

    let internal toColor (c : string) (vertices : seq<string>) =
        if vertices |> Seq.isEmpty then "" else
        let formatstr = 
            let f = "{0} [style=filled, color={1}"
            if c = "blue" then f + ", fontcolor=white]" else f + "]"
        vertices
        |> Seq.map (fun v -> String.Format(formatstr, v, c))
        |> Seq.reduce (+)
                

    let internal coloring (self: GraphSeq) (selfRev : GraphSeq) in' out =
        let bottomOutgoing (graphSeq : seq<string * string []>) bottom = 
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
    let internal visualizeSubgraph (subgraph : GraphSeq) (subgraphRev : GraphSeq) outConMin inConMin clusterN =
        let graphOpen = if clusterN >= 0 then "subgraph cluster_" + clusterN.ToString() + "{ color = red; " else "digraph {"
        let graphClose = "}"

        let colorOut, colorIn, colorBoth = coloring subgraph subgraphRev inConMin outConMin

        let visualizable = 
            subgraph 
            |> Seq.map 
                (fun (v, c) -> 
                    if c.Length = 0 then v
                    else
                    c 
                    |> Array.map (fun s -> v + " -> " + s)
                    |> Array.reduce (fun acc e -> acc + "; " + e))
            |> Seq.reduce (fun acc e -> acc + "; " + e)
            |> fun v -> graphOpen + colorOut + colorIn + colorBoth + v + graphClose

        visualizable

    let internal visualizeEntire subgraph subgraphRev outConMin inConMin = 
        visualizeSubgraph subgraph subgraphRev outConMin inConMin  -1 
        |> createVisual
        
    let internal visualizeAll (graph : DirectedGraph) outConMin inConMin clusters euler eulerLabels =
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
            |> fun gr -> eulerPath.[0] + "[color=green, style=filled]; " + gr
            |> fun gr -> createVisualClusters ("digraph { " + gr + "}") 

        else 
            let rev = graph.Reverse
            let self = graph.AsEnumerable
            let selfRev = rev.AsEnumerable

            let connectedComponents = 
                if clusters then 
                    graph.FindConnectedComponents() 
                    |> List.map (fun h -> h.AsEnumerable() |> Seq.toList) 
                else []

            if not clusters then visualizeEntire self selfRev outConMin inConMin
            else
                connectedComponents 
                |> List.mapi
                    (fun i vertices ->
                        let subgraph = vertices |> graph.Subgraph
                        let subgraphRev = vertices |> rev.Subgraph
                        visualizeSubgraph subgraph subgraphRev outConMin inConMin i
                    )
                |> List.reduce (+)
                |> fun gr -> createVisualClusters ("digraph { " + gr + "}") 
                            
    type Visualizer () =
        /// <summary>
        /// Visualize the graph. Should in/out connections be emphasized
        /// </summary>
        /// <param name="into">Optional. If present - should be the minimum number of inbound connections which would select the vertex for coloring.</param>
        /// <param name="out">Optional. If present - should be the minimum number of outbound connections which would select the vertex for coloring.</param>
        static member Visualize(graph : DirectedGraph, ?into, ?out, ?clusters, ?euler, ?eulerLabels : string seq) =
            let outConMin = defaultArg out 0
            let inConMin = defaultArg into 0
            let clusters = defaultArg clusters false
            let euler = defaultArg euler false
            let eulerLabels = defaultArg eulerLabels Seq.empty

            visualizeAll graph outConMin inConMin clusters euler eulerLabels
