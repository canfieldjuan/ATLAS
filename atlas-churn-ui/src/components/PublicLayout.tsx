import { Link, useLocation } from 'react-router-dom'
import { Activity } from 'lucide-react'

const NAV_LINKS = [
  { label: 'Blog', to: '/blog' },
  { label: 'Dashboard', to: '/' },
]

export default function PublicLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col">
      {/* Nav */}
      <nav className="flex items-center justify-between max-w-6xl mx-auto w-full px-6 py-5">
        <div className="flex items-center gap-6">
          <Link to="/blog" className="flex items-center gap-2">
            <Activity className="h-7 w-7 text-cyan-400" />
            <span className="text-xl font-bold">Churn Intel</span>
          </Link>
          {NAV_LINKS.map(link => (
            <Link
              key={link.to}
              to={link.to}
              className={`text-sm font-medium transition-colors ${
                location.pathname.startsWith(link.to) && link.to !== '/'
                  ? 'text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </nav>

      {/* Page content */}
      <main className="flex-1">{children}</main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-8">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-sm text-slate-500">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-slate-600" />
            <span>Churn Intel</span>
          </div>
          <div className="flex items-center gap-4">
            <Link to="/blog" className="hover:text-slate-300 transition-colors">Blog</Link>
            <Link to="/" className="hover:text-slate-300 transition-colors">Dashboard</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
