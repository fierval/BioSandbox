#I @"C:\Git\GraphViz-C-Sharp-Wrapper\src\GraphVizWrapper\bin\Debug"
#r @"GraphVizWrapper.dll"

open GraphVizWrapper
open GraphVizWrapper.Queries
open GraphVizWrapper.Commands

let getStartProcessQuery = GetStartProcessQuery();
let getProcessStartInfoQuery = GetProcessStartInfoQuery();
let registerLayoutPluginCommand = RegisterLayoutPluginCommand(getProcessStartInfoQuery, getStartProcessQuery);

// GraphGeneration can be injected via the IGraphGeneration interface

let wrapper = GraphGeneration(getStartProcessQuery, 
                                  getProcessStartInfoQuery, 
                                  registerLayoutPluginCommand);

wrapper.GraphvizPath <- @"C:\Program Files (x86)\Graphviz2.38\bin"

let output = wrapper.GenerateGraph("digraph{a -> b; b -> c; c -> a}", Enums.GraphReturnType.Png)