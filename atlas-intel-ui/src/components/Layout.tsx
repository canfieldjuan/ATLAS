import { useState, type ReactNode } from 'react'
import { Menu, Search } from 'lucide-react'
import Sidebar from './Sidebar'

export default function Layout({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen">
      {/* Mobile header */}
      <header className="fixed top-0 left-0 right-0 h-14 bg-slate-900/80 border-b border-slate-700/50 backdrop-blur flex items-center px-4 gap-3 z-10 lg:hidden">
        <button
          onClick={() => setSidebarOpen(true)}
          className="text-slate-400 hover:text-white"
        >
          <Menu className="h-5 w-5" />
        </button>
        <Search className="h-5 w-5 text-cyan-400" />
        <span className="text-sm font-semibold text-white">Consumer Intel</span>
      </header>

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <main className="lg:ml-56 p-6 pt-20 lg:pt-6">{children}</main>
    </div>
  )
}
