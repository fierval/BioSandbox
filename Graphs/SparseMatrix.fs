namespace Graphs
open System
open Alea.CUDA
open Alea.CUDA.Unbound
open Microsoft.FSharp.Math
open System.Collections.Generic
open System.Linq

[<AutoOpen>]
module SparseMatrixModule = 

    type SparseMatrix<'a> (ops : INumeric<'a>, row : 'a seq, rowIndex : int seq, colIndex : int seq, rowSize, isCSR : bool) =

        let isCSR = isCSR
        let ops = ops
        let mutable rowIndex  = rowIndex.ToList()
        let mutable colIndex = colIndex.ToList()
        let mutable values = row.ToList()
        let mutable nnz = rowIndex.[rowIndex.Count - 1]
        let rowSize = rowSize

        let getRowCol (row, col) =
            if row >= rowIndex.Count - 1 then 
                failwith ( if isCSR then "row index out of range" else "column index out of range")

            if col >= rowSize then failwith ( if isCSR then "column index out of range" else "row index out of range")

            let rowStart = rowIndex.[row]
            let rowEnd = rowIndex.[row + 1] - 1


            let valCol = colIndex.GetRange(rowStart, rowEnd - rowStart + 1).IndexOf(col)
        
            if valCol >= 0 then values.[rowStart + valCol] else ops.Zero
        
        member this.Item
            with get(row, col) = 
                if isCSR then getRowCol(row, col) else getRowCol(col, row)

        member this.NNZ = nnz

        member this.AddValues (row : 'a []) =
            if row.Length <> rowSize then failwith "wrong number of elements in the row/column"
            let colIdx, vals = 
                Array.zip [|0..row.Length - 1|] row 
                |> Array.filter (fun (i, v) -> ops.Compare(v, ops.Zero) <> 0)
                |> Array.unzip
            
            values.AddRange(vals)
            colIndex.AddRange(colIdx)

            rowIndex.Add(rowIndex.[rowIndex.Count - 1] + vals.Length)
            nnz <- nnz + vals.Length

        member this.PrintMatrix () =
            let rows, cols = if isCSR then rowIndex.Count - 1, rowSize else rowSize, rowIndex.Count - 1

            let printRows, printCols = (if rows > 10 then 10 else rows), if cols > 10 then 10 else cols
            let elipses = if rows > 10 || cols > 10 then "..." else ""

            for i = 0 to printRows - 1 do
              let str = sprintf "%A%s" [for j = 0 to printCols - 1 do yield this.[i, j]] elipses
              printfn "%s" str

    let createMatrix (row : 'a []) (isCSR : bool) =
        let ops = GlobalAssociations.GetNumericAssociation<'a>()
        let colIdx, vals = 
            Array.zip [|0..row.Length - 1|] row 
            |> Array.filter (fun (i, v) -> ops.Compare(v, ops.Zero) <> 0)
            |> Array.unzip

        SparseMatrix(ops, vals, [0; vals.Length], colIdx, row.Length, isCSR)

           