import { Link, useLocation } from 'react-router-dom'
import AtlasRobotLogo from './AtlasRobotLogo'
import { buildLoginRedirectPath, buildSignupRedirectPath } from '../auth/redirects'

const NAV_LINKS = [
  { label: 'Blog', to: '/blog' },
  { label: 'Methodology', to: '/methodology' },
]

const DEFAULT_LOGIN_TO = buildLoginRedirectPath('/watchlists', 'b2b_retention')
const DEFAULT_SIGNUP_TO = buildSignupRedirectPath('/watchlists', 'b2b_retention')

type Props = {
  children: React.ReactNode
  /** "report" strips Blog/Sign-in, changes CTA to "Get Weekly Intelligence" */
  variant?: 'default' | 'report'
  /** Called when the report-variant CTA is clicked (wires to Stripe checkout) */
  onCtaClick?: () => void
}

export default function PublicLayout({ children, variant = 'default', onCtaClick }: Props) {
  const location = useLocation()
  const isReport = variant === 'report'

  const reportCta = isReport && onCtaClick ? (
    <button
      onClick={onCtaClick}
      className="text-sm px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors cursor-pointer"
    >
      Get Weekly Intelligence
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
            <Link to={DEFAULT_LOGIN_TO} className="text-sm text-slate-400 hover:text-white transition-colors">
              Sign in
            </Link>
          )}
          {reportCta}
          {!isReport && !reportCta && (
            <Link
              to={DEFAULT_SIGNUP_TO}
              className="text-sm px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors"
            >
              Start Free Trial
            </Link>
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
                <Link to="/methodology" className="hover:text-slate-300 transition-colors">Methodology</Link>
                <Link to={DEFAULT_LOGIN_TO} className="hover:text-slate-300 transition-colors">Sign in</Link>
              </>
            )}
            {isReport ? (
              onCtaClick && (
                <button onClick={onCtaClick} className="hover:text-slate-300 transition-colors cursor-pointer">
                  Get Weekly Intelligence
                </button>
              )
            ) : (
              <Link to={DEFAULT_SIGNUP_TO} className="hover:text-slate-300 transition-colors">Sign up</Link>
            )}
          </div>
        </div>
      </footer>
    </div>
  )
}
