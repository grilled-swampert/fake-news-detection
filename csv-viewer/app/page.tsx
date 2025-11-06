"use client"

import { useState } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import CSVUploader from "@/components/csv-uploader"
import CSVDisplay from "@/components/csv-display"
import LanguageSwitcher from "@/components/language-switcher"

const translations = {
  en: {
    title: "CSV Viewer",
    subtitle: "Upload and display your data with ease",
    upload: "Upload CSV",
    noFile: "No CSV file uploaded yet",
    selectLanguage: "Language",
  },
  es: {
    title: "Visor CSV",
    subtitle: "Carga y muestra tus datos fácilmente",
    upload: "Cargar CSV",
    noFile: "No hay archivo CSV cargado aún",
    selectLanguage: "Idioma",
  },
  fr: {
    title: "Visionneuse CSV",
    subtitle: "Téléchargez et affichez vos données facilement",
    upload: "Télécharger CSV",
    noFile: "Aucun fichier CSV chargé pour le moment",
    selectLanguage: "Langue",
  },
  pt: {
    title: "Visualizador CSV",
    subtitle: "Envie e exiba seus dados facilmente",
    upload: "Enviar CSV",
    noFile: "Nenhum arquivo CSV enviado ainda",
    selectLanguage: "Idioma",
  },
  it: {
    title: "Visualizzatore CSV",
    subtitle: "Carica e visualizza i tuoi dati con facilità",
    upload: "Carica CSV",
    noFile: "Nessun file CSV caricato ancora",
    selectLanguage: "Lingua",
  },
}

export default function Home() {
  const [csvData, setCsvData] = useState<any[] | null>(null)
  const [fileName, setFileName] = useState<string>("")
  const [language, setLanguage] = useState<keyof typeof translations>("en")

  const t = translations[language]

  const handleFileUpload = (data: any[], name: string) => {
    setCsvData(data)
    setFileName(name)
  }

  const handleClear = () => {
    setCsvData(null)
    setFileName("")
  }

  return (
    <main className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">{t.title}</h1>
            <p className="mt-1 text-base text-muted-foreground">{t.subtitle}</p>
          </div>
          <LanguageSwitcher currentLanguage={language} onLanguageChange={setLanguage} />
        </div>
      </header>

      {/* Main Content - Full Page Display */}
      {!csvData ? (
        <div className="flex flex-1 items-center justify-center p-8">
          <Card className="w-full max-w-lg">
            <div className="p-12">
              <h2 className="mb-8 text-center text-2xl font-medium text-foreground">{t.upload}</h2>
              <CSVUploader onFileUpload={handleFileUpload} />
            </div>
          </Card>
        </div>
      ) : (
        <div className="flex flex-1 flex-col overflow-hidden p-8">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Current File:</p>
              <p className="text-lg font-medium text-foreground">{fileName}</p>
            </div>
            <Button variant="outline" onClick={handleClear} className="gap-2 bg-transparent">
              <X className="h-4 w-4" />
              Clear
            </Button>
          </div>
          <div className="flex-1 overflow-hidden">
            <CSVDisplay data={csvData} fileName={fileName} />
          </div>
        </div>
      )}
    </main>
  )
}
