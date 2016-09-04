namespace GpuEuler

    [<AutoOpen>]
    module Swips =
        open Graphs

        /// <summary>
        /// Compute the actual edges that need to be swapped to create a single loop
        /// </summary>
        /// <param name="gr">Partitioned graph</param>
        /// <param name="links">A map E(partitioned) -> E(originalGraph) </param>
        /// <param name="predecessors">The original array of predecessors</param>
        /// <param name="validity">"Validity" array (output of the previous step)</param>
        let fixPredecessors (gr : DirectedGraph<'a>) (links : int[]) (predecessors : int []) (validity : bool []) =

            gr.SpanningTreeEdges
            |> Array.map (fun e -> links.[e])
            |> Array.iter( fun s ->
                let mutable j = s + 1
                while not validity.[j] do
                    j <- j + 1

                // swap predecessors
                let temp = predecessors.[s]
                predecessors.[s] <- predecessors.[j]
                predecessors.[j] <- temp

            )

            predecessors
