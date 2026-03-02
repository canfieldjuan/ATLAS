import { useState, type FormEvent } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { Search, AlertCircle, ShieldCheck, Target, Users } from 'lucide-react'
import { clsx } from 'clsx'
import { useAuth } from '../auth/AuthContext'

const PRODUCTS = [
  {
    id: 'consumer',
    label: 'Amazon Seller Intel',
    desc: 'Track competitor products, complaints, and feature gaps',
    icon: Search,
    color: 'cyan',
  },
  {
    id: 'b2b_retention',
    label: 'Vendor Retention',
    desc: 'Track YOUR vendors to see churn signals and pain trends',
    icon: ShieldCheck,
    color: 'violet',
  },
  {
    id: 'b2b_challenger',
    label: 'Challenger Lead Gen',
    desc: 'Track COMPETITOR vendors to find high-intent leads',
    icon: Target,
    color: 'amber',
  },
] as const

export default function Signup() {
  const { user, signup } = useAuth()
  const navigate = useNavigate()
  const [product, setProduct] = useState<string>('consumer')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [accountName, setAccountName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  if (user) {
    const home = user.product === 'consumer' ? '/' : '/b2b'
    return <Navigate to={home} replace />
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError('')
    if (password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }
    setLoading(true)
    try {
      await signup(email, password, fullName, accountName, product)
      const dest = product === 'consumer' ? '/onboarding' : '/b2b/onboarding'
      navigate(dest)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-lg">
        <div className="flex items-center justify-center gap-2 mb-8">
          <Users className="h-8 w-8 text-cyan-400" />
          <span className="text-2xl font-bold text-white">Atlas Intelligence</span>
        </div>

        <form onSubmit={handleSubmit} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 space-y-5">
          <h2 className="text-lg font-semibold text-white text-center">Create your account</h2>

          {error && (
            <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}

          {/* Product selector */}
          <div className="space-y-2">
            <label className="block text-sm text-slate-400">Choose your product</label>
            <div className="grid grid-cols-1 gap-2">
              {PRODUCTS.map(p => {
                const Icon = p.icon
                const selected = product === p.id
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => setProduct(p.id)}
                    className={clsx(
                      'flex items-start gap-3 p-3 rounded-lg border text-left transition-colors',
                      selected
                        ? `border-${p.color}-500/50 bg-${p.color}-900/20`
                        : 'border-slate-700/50 bg-slate-900/40 hover:border-slate-600/50',
                    )}
                    style={selected ? {
                      borderColor: p.color === 'cyan' ? 'rgb(6 182 212 / 0.5)' : p.color === 'violet' ? 'rgb(139 92 246 / 0.5)' : 'rgb(245 158 11 / 0.5)',
                      backgroundColor: p.color === 'cyan' ? 'rgb(22 78 99 / 0.2)' : p.color === 'violet' ? 'rgb(76 29 149 / 0.2)' : 'rgb(120 53 15 / 0.2)',
                    } : undefined}
                  >
                    <Icon className={clsx('h-5 w-5 mt-0.5 shrink-0', selected ? 'text-white' : 'text-slate-500')} />
                    <div>
                      <div className={clsx('text-sm font-medium', selected ? 'text-white' : 'text-slate-300')}>
                        {p.label}
                      </div>
                      <div className="text-xs text-slate-500">{p.desc}</div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Full name</label>
            <input
              type="text"
              value={fullName}
              onChange={e => setFullName(e.target.value)}
              required
              className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              placeholder="Jane Smith"
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Company name</label>
            <input
              type="text"
              value={accountName}
              onChange={e => setAccountName(e.target.value)}
              required
              className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              placeholder="Acme Inc."
            />
          </div>

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

          <div>
            <label className="block text-sm text-slate-400 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              minLength={8}
              className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              placeholder="Min. 8 characters"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded-lg text-white font-medium transition-colors"
          >
            {loading ? 'Creating account...' : 'Create account'}
          </button>

          <p className="text-center text-sm text-slate-400">
            Already have an account?{' '}
            <Link to="/login" className="text-cyan-400 hover:text-cyan-300">
              Sign in
            </Link>
          </p>
        </form>

        <p className="mt-4 text-center text-xs text-slate-500">
          14-day free trial. No credit card required.
        </p>
      </div>
    </div>
  )
}
