"use client"

import { useState } from "react"
import { ChevronLeft, ChevronRight, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface CSVDisplayProps {
  data: any[]
  fileName: string
}

export default function CSVDisplay({ data, fileName }: CSVDisplayProps) {
  const [currentPage, setCurrentPage] = useState(0)
  const rowsPerPage = 20

  const columns = data.length > 0 ? Object.keys(data[0]) : []
  const totalPages = Math.ceil(data.length / rowsPerPage)
  const start = currentPage * rowsPerPage
  const end = start + rowsPerPage
  const currentData = data.slice(start, end)

  const downloadCSV = () => {
    const csv = [columns.join(","), ...data.map((row) => columns.map((col) => `"${row[col]}"`).join(","))].join("\n")

    const blob = new Blob([csv], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = fileName || "data.csv"
    a.click()
  }

  if (data.length === 0 || columns.length === 0) {
    return null
  }

  return (
    <Card className="flex h-full flex-col bg-card rounded-lg border border-border overflow-hidden">
      <div className="flex items-center justify-between border-b border-border bg-muted/20 px-8 py-6">
        <div>
          <h3 className="text-2xl font-semibold text-foreground">Data Preview</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Showing {start + 1} to {Math.min(end, data.length)} of {data.length} rows
          </p>
        </div>
        <Button onClick={downloadCSV} variant="outline" className="gap-2 bg-transparent">
          <Download className="h-4 w-4" />
          Download
        </Button>
      </div>

      <div className="flex-1 overflow-auto px-8 py-6">
        <div className="rounded-lg border border-border overflow-hidden">
          <table className="w-full text-base">
            <thead>
              <tr className="border-b border-border bg-muted/30 sticky top-0">
                {columns.map((col) => (
                  <th key={col} className="px-6 py-4 text-left font-semibold text-foreground whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {currentData.map((row, idx) => (
                <tr key={idx} className="border-b border-border hover:bg-muted/40 transition-colors">
                  {columns.map((col) => (
                    <td key={`${idx}-${col}`} className="px-6 py-4 text-foreground">
                      {row[col]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-border bg-muted/20 px-8 py-6">
          <p className="text-sm text-muted-foreground">
            Page {currentPage + 1} of {totalPages}
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
              disabled={currentPage === 0}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={currentPage === totalPages - 1}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}
