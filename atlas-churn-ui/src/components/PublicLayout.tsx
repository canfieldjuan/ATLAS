import { Link, useLocation } from 'react-router-dom'
import AtlasRobotLogo from './AtlasRobotLogo'

const NAV_LINKS = [
  { label: 'Blog', to: '/blog' },
]

type Props = {
  children: React.ReactNode
  /** "report" strips Blog/Sign-in, changes CTA to "Get Weekly Reports" */
  variant?: 'default' | 'report'
  /** Called when the report-variant CTA is clicked (wires to Stripe checkout) */
  onCtaClick?: () => void
}

export default function PublicLayout({ children, variant = 'default', onCtaClick }: Props) {
  const location = useLocation()
  const isReport = variant === 'report'

  const reportCta = isReport ? (
    <button
      onClick={onCtaClick}
      className="text-sm px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors cursor-pointer"
    >
      Get Weekly Reports
    </button>
  ) : null

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col">
      {/* Nav */}
      <nav className="flex items-center justify-between max-w-6xl mx-auto w-full px-6 py-5">
        <div className="flex items-center gap-6">
          <Link to="/landing" className="flex items-center gap-2">
            <AtlasRobotLogo className="h-7 w-7" />
            <span className="text-xl font-bold">Churn Signals</span>
          </Link>
          {!isReport && NAV_LINKS.map(link => (
            <Link
              key={link.to}
              to={link.to}
              className={`text-sm font-medium transition-colors ${
                location.pathname.startsWith(link.to)
                  ? 'text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
        <div className="flex items-center gap-4">
          {!isReport && (
            <Link to="/login" className="text-sm text-slate-400 hover:text-white transition-colors">
              Sign in
            </Link>
          )}
          {reportCta || (
            <a
              href="/signup"
              className="text-sm px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors"
            >
              Start Free Trial
            </a>
          )}
        </div>
      </nav>

      {/* Page content */}
      <main className="flex-1">{children}</main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-8">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-sm text-slate-500">
          <div className="flex items-center gap-2">
            <AtlasRobotLogo className="h-4 w-4" />
            <span>Churn Signals</span>
          </div>
          <div className="flex items-center gap-4">
            {!isReport && (
              <>
                <Link to="/blog" className="hover:text-slate-300 transition-colors">Blog</Link>
                <Link to="/login" className="hover:text-slate-300 transition-colors">Sign in</Link>
              </>
            )}
            {isReport ? (
              <button onClick={onCtaClick} className="hover:text-slate-300 transition-colors cursor-pointer">
                Get Weekly Reports
              </button>
            ) : (
              <a href="/signup" className="hover:text-slate-300 transition-colors">Sign up</a>
            )}
          </div>
        </div>
      </footer>
    </div>
  )
}
