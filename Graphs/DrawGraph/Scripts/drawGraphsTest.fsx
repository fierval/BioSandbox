open System
open System.IO

Environment.CurrentDirectory <- Path.Combine(__SOURCE_DIRECTORY__, @"../bin/Release/x64")

#load "load-project-release.fsx"
open DrawGraph

createGraph "digraph{video->extract_1; video->extract_2; video->extract_3; extract_1->save_1; extract_2->save_2; extract_3->save_3}" "dot.exe" None

