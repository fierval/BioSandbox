// Learn more about F# at http://fsharp.org. See the 'F# Tutorial' project
// for more guidance on F# programming.
#r @"..\..\packages\FsCheck.2.4.0\lib\net45\FsCheck.dll"
#r @"C:\Emgu\emgucv-windesktop_x64-cuda 3.1.0.2282\bin\Emgu.CV.UI.dll"
#r @"C:\Emgu\emgucv-windesktop_x64-cuda 3.1.0.2282\bin\Emgu.CV.World.dll"
#r @"System.Drawing"
#r @"System.Windows.Forms"

#load "drawGraphs.fs"
open DrawGraph

createGraph "digraph{a->b; b->c; 2->1; d->b; b->b; a->d}" "dot.exe" None

// Define your library scripting code here

