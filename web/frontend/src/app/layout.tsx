import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Sidebar } from "@/components/layout/sidebar"
import { Header } from "@/components/layout/header"
import { TooltipProvider } from "@/components/ui/tooltip"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "TradingAgents",
  description: "Multi-agent AI trading system",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <TooltipProvider>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div className="flex flex-1 flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-auto p-6">{children}</main>
            </div>
          </div>
        </TooltipProvider>
      </body>
    </html>
  )
}
