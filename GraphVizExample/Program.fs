open GraphVizWrapper
open GraphVizWrapper.Commands
open GraphVizWrapper.Queries
open Emgu.CV
open Emgu.CV.Structure
open System.IO
open System.Drawing

[<EntryPoint>]
let main argv = 
    let getStartProcessQuery = GetStartProcessQuery();
    let getProcessStartInfoQuery = GetProcessStartInfoQuery();
    let registerLayoutPluginCommand = RegisterLayoutPluginCommand(getProcessStartInfoQuery, getStartProcessQuery);

    // GraphGeneration can be injected via the IGraphGeneration interface

    let wrapper = GraphGeneration(getStartProcessQuery, 
                                      getProcessStartInfoQuery, 
                                      registerLayoutPluginCommand);

    wrapper.GraphvizPath <- @"C:\Program Files (x86)\Graphviz2.38\bin"
    wrapper.RenderingEngine <- Enums.RenderingEngine.Neato

    let output = wrapper.GenerateGraph("digraph{a -> b; b -> c; c -> a}", Enums.GraphReturnType.Png)

    let ms = new MemoryStream(output);
    use bmpImage : Bitmap = new Bitmap(System.Drawing.Image.FromStream(ms))
    use im = new Image<Bgr, byte>(bmpImage)

    CvInvoke.Imshow("Graph", im)
    CvInvoke.WaitKey(0) |> ignore
    0 // return an integer exit code
