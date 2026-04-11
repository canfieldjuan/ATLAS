import { useState, type FormEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { AlertCircle, CheckCircle, ArrowLeft } from 'lucide-react'
import { inferRedirectProduct, normalizeRedirectTarget } from '../auth/redirects'
import AtlasRobotLogo from '../components/AtlasRobotLogo'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export default function ForgotPassword() {
  const [searchParams] = useSearchParams()
  const rawRedirectTo = searchParams.get('redirect_to')
  const redirectTo = rawRedirectTo ? normalizeRedirectTarget(rawRedirectTo) : ''
  const product = searchParams.get('product')?.trim() || (redirectTo ? inferRedirectProduct(redirectTo) : '')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [sent, setSent] = useState(false)
  const [loading, setLoading] = useState(false)

  const loginParams = new URLSearchParams()
  if (redirectTo) loginParams.set('redirect_to', redirectTo)
  if (product) loginParams.set('product', product)
  const loginHref = loginParams.toString() ? `/login?${loginParams.toString()}` : '/login'

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Request failed' }))
        throw new Error(body.detail || `Error ${res.status}`)
      }
      setSent(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="flex items-center justify-center gap-2 mb-8">
          <AtlasRobotLogo className="h-8 w-8" />
          <span className="text-2xl font-bold text-white">Churn Signals</span>
        </div>

        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 space-y-4">
          {sent ? (
            <>
              <div className="flex items-center gap-2 text-sm text-green-400 bg-green-900/20 border border-green-800/50 rounded-lg px-3 py-2">
                <CheckCircle className="h-4 w-4 shrink-0" />
                Check your email for a reset link
              </div>
              <p className="text-sm text-slate-400 text-center">
                If an account exists for <span className="text-white">{email}</span>,
                you'll receive a password reset link within a few minutes.
              </p>
              <Link
                to={loginHref}
                className="flex items-center justify-center gap-2 text-sm text-cyan-400 hover:text-cyan-300 mt-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to sign in
              </Link>
            </>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <h2 className="text-lg font-semibold text-white text-center">Reset your password</h2>
              <p className="text-sm text-slate-400 text-center">
                Enter your email and we'll send you a link to reset your password.
              </p>

              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  {error}
                </div>
              )}

              <div>
                <label className="block text-sm text-slate-400 mb-1">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  placeholder="you@company.com"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded-lg text-white font-medium transition-colors"
              >
                {loading ? 'Sending...' : 'Send reset link'}
              </button>

              <Link
                to={loginHref}
                className="flex items-center justify-center gap-2 text-sm text-cyan-400 hover:text-cyan-300"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to sign in
              </Link>
            </form>
          )}
        </div>
      </div>
    </div>
  )
}
