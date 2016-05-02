﻿module DrawGraph

open System
open System.Diagnostics
open System.IO
open Emgu.CV
open Emgu.CV.UI
open Emgu.CV.CvEnum

let createGraph (graph : string) (graphVizPath : string option) =
    let workingDir = 
        match graphVizPath with
        | Some p -> p
        | None -> String.Empty

    let graphFile = Path.GetTempFileName()
    File.WriteAllText(graphFile, graph)

    let pi = ProcessStartInfo(Path.Combine(workingDir, "dot.exe"))
    pi.CreateNoWindow <- true
    pi.ErrorDialog <- false;
    pi.UseShellExecute <- false;
    pi.Arguments <- String.Format("-Tpng -O {0}", graphFile)
    pi.WorkingDirectory <- workingDir
    try
        let proc = new Process();
        proc.StartInfo <- pi
        proc.Start() |> ignore

        try
            proc.WaitForExit()
        with
        | _ -> ()

        if proc.ExitCode = 0 then
            let mat = CvInvoke.Imread(graphFile + ".png", LoadImageType.AnyColor)
            let viewer = new ImageViewer(mat)
            viewer.Show()
        else failwith "could not create image file"                  
    finally
        if File.Exists graphFile then File.Delete graphFile
        if File.Exists (graphFile + ".png") then File.Delete (graphFile + ".png")