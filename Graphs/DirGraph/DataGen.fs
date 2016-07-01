module DataGen
open FsCheck
open System
open Graphs

//let nucl = Gen.oneof [gen {return 'A'}; gen {return 'C'}; gen {return 'G'}; gen {return 'T'}]
let nucl = Gen.choose(int 'A', int 'Z') |> Gen.map char

let genVertex len =  Gen.arrayOfLength len nucl |> Gen.map (fun c -> String(c))
let vertices len number = Gen.arrayOfLength number (genVertex len) |> Gen.map Array.distinct

let graphGen len number =
    let verts = vertices len number
    let rnd = Random(int DateTime.UtcNow.Ticks)
    let pickFrom = verts |> Gen.map (fun lst -> lst.[rnd.Next(lst.Length)])
    let pickTo = Gen.sized (fun n -> Gen.listOfLength (if n = 0 then 1 else n) pickFrom)

    Gen.sized
    <| 
    (fun n ->
        Gen.map2 
            (fun from to' -> 
                from, (to' |> Seq.reduce (fun acc v -> acc + ", " + v))) pickFrom pickTo
        |>
        Gen.arrayOfLength (if n = 0 then 1 else n)
        |> Gen.map (Array.distinctBy fst)
        |> Gen.map (fun arr ->  arr |> Array.map (fun (a, b) -> a + " -> " + b))
    )
    |> Gen.map StrGraph.FromStrings

