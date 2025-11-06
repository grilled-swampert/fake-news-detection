"use client"

import type React from "react"

import { useCallback, useState } from "react"
import { Upload } from "lucide-react"
import Papa from "papaparse"

interface CSVUploaderProps {
  onFileUpload: (data: any[], fileName: string) => void
}

export default function CSVUploader({ onFileUpload }: CSVUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleParse = (file: File) => {
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        onFileUpload(
          results.data.filter((row: any) => Object.values(row).some((val) => val)),
          file.name,
        )
      },
      error: (error) => {
        console.error("CSV parsing error:", error)
      },
    })
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type === "text/csv") {
      handleParse(file)
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleParse(file)
    }
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`relative rounded-lg border-2 border-dashed transition-colors ${
        isDragging ? "border-primary bg-primary/5" : "border-border bg-muted/30 hover:border-primary/50"
      }`}
    >
      <label className="flex cursor-pointer flex-col items-center justify-center p-8">
        <Upload className="mb-2 h-8 w-8 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Drop your CSV here</span>
        <span className="mt-1 text-xs text-muted-foreground">or click to browse</span>
        <input type="file" accept=".csv" onChange={handleFileSelect} className="hidden" />
      </label>
    </div>
  )
}
