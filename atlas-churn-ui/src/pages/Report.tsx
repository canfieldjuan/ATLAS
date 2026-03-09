import { useState, useEffect, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { FileText, Mail, CheckCircle, AlertCircle, Loader2, ShieldCheck } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const GATE_URL = `${API_BASE}/api/v1/b2b/briefings/gate`

type Status = 'idle' | 'submitting' | 'success' | 'error'

export default function Report() {
  const [params] = useSearchParams()
  const vendor = params.get('vendor') || ''
  const token = params.get('ref') || ''

  const [email, setEmail] = useState('')
  const [status, setStatus] = useState<Status>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    document.title = vendor
      ? `${vendor} Churn Intelligence Report -- Churn Signals`
      : 'Churn Intelligence Report -- Churn Signals'
  }, [vendor])

  if (!vendor || !token) {
    return (
      <PublicLayout>
        <div className="min-h-[60vh] flex items-center justify-center px-6">
          <div className="max-w-md text-center">
            <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
            <h1 className="text-2xl font-bold mb-2">Invalid Link</h1>
            <p className="text-slate-400">
              This report link is missing required parameters. Check the link from your email and try again.
            </p>
          </div>
        </div>
      </PublicLayout>
    )
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!email.trim()) return

    setStatus('submitting')
    setErrorMsg('')

    try {
      const res = await fetch(GATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), token }),
      })

      if (res.ok) {
        setStatus('success')
      } else {
        const body = await res.json().catch(() => ({ detail: 'Something went wrong' }))
        setErrorMsg(body.detail || `Error ${res.status}`)
        setStatus('error')
      }
    } catch {
      setErrorMsg('Network error -- please try again')
      setStatus('error')
    }
  }

  if (status === 'success') {
    return (
      <PublicLayout>
        <div className="min-h-[60vh] flex items-center justify-center px-6">
          <div className="max-w-lg text-center">
            <CheckCircle className="h-16 w-16 text-emerald-400 mx-auto mb-6" />
            <h1 className="text-3xl font-bold mb-3">Check your inbox</h1>
            <p className="text-lg text-slate-400 mb-2">
              The full <span className="text-white font-semibold">{vendor}</span> churn intelligence report is on its way.
            </p>
            <p className="text-sm text-slate-500">
              Sent to <span className="text-slate-300">{email}</span> -- usually arrives within 60 seconds.
            </p>
          </div>
        </div>
      </PublicLayout>
    )
  }

  return (
    <PublicLayout>
      <div className="min-h-[60vh] flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-md">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-5">
              <FileText className="h-7 w-7 text-cyan-400" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold mb-3">
              <span className="text-cyan-400">{vendor}</span> Churn Intelligence
            </h1>
            <p className="text-slate-400 leading-relaxed">
              Enter your work email to receive the full report -- account-level churn signals, displacement data, and risk scores.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
              <input
                type="email"
                required
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@company.com"
                autoComplete="email"
                className="w-full pl-11 pr-4 py-3 bg-slate-800/80 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-colors"
              />
            </div>

            <button
              type="submit"
              disabled={status === 'submitting' || !email.trim()}
              className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-white font-semibold transition-colors flex items-center justify-center gap-2"
            >
              {status === 'submitting' ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Sending report...
                </>
              ) : (
                'Send me the full report'
              )}
            </button>

            {status === 'error' && (
              <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
                <p className="text-sm text-red-300">{errorMsg}</p>
              </div>
            )}
          </form>

          {/* Trust signals */}
          <div className="mt-8 flex items-center justify-center gap-2 text-xs text-slate-500">
            <ShieldCheck className="h-4 w-4" />
            <span>No spam. Unsubscribe anytime. Data sourced from public reviews.</span>
          </div>
        </div>
      </div>
    </PublicLayout>
  )
}
