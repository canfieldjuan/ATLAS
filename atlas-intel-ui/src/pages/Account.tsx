import { useState } from 'react'
import { User, CreditCard, Settings, LogOut, ExternalLink, AlertCircle } from 'lucide-react'
import { useAuth } from '../auth/AuthContext'

const PLAN_LABELS: Record<string, string> = {
  trial: 'Trial',
  starter: 'Starter',
  growth: 'Growth',
  pro: 'Pro',
}

const PLAN_COLORS: Record<string, string> = {
  trial: 'bg-slate-700 text-slate-300',
  starter: 'bg-cyan-900/50 text-cyan-300',
  growth: 'bg-violet-900/50 text-violet-300',
  pro: 'bg-amber-900/50 text-amber-300',
}

export default function Account() {
  const { user, logout } = useAuth()
  const [portalLoading, setPortalLoading] = useState(false)
  const [checkoutLoading, setCheckoutLoading] = useState(false)
  const [error, setError] = useState('')

  if (!user) return null

  const planBadge = PLAN_COLORS[user.plan] || PLAN_COLORS.trial

  function check401(res: Response): boolean {
    if (res.status === 401) {
      localStorage.removeItem('atlas_token')
      localStorage.removeItem('atlas_refresh_token')
      window.location.href = '/login'
      return true
    }
    return false
  }

  async function openBillingPortal() {
    setPortalLoading(true)
    setError('')
    try {
      const token = localStorage.getItem('atlas_token')
      const res = await fetch('/api/v1/billing/portal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      })
      if (check401(res)) return
      if (!res.ok) throw new Error('Failed to open billing portal')
      const data = await res.json()
      window.location.href = data.portal_url
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open billing portal')
    } finally {
      setPortalLoading(false)
    }
  }

  async function startCheckout(planName: string) {
    setCheckoutLoading(true)
    setError('')
    try {
      const token = localStorage.getItem('atlas_token')
      const res = await fetch('/api/v1/billing/checkout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          plan: planName,
          success_url: `${window.location.origin}/account?upgraded=1`,
          cancel_url: `${window.location.origin}/account`,
        }),
      })
      if (check401(res)) return
      if (!res.ok) throw new Error('Failed to start checkout')
      const data = await res.json()
      window.location.href = data.checkout_url
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start checkout')
    } finally {
      setCheckoutLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">Account</h1>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Profile */}
      <section className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-cyan-900/50 flex items-center justify-center">
            <User className="h-5 w-5 text-cyan-400" />
          </div>
          <div>
            <div className="text-white font-medium">{user.full_name || user.email}</div>
            <div className="text-sm text-slate-400">{user.email}</div>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-400">
          <Settings className="h-4 w-4" />
          <span>Account: {user.account_name}</span>
          <span className="text-slate-600">|</span>
          <span>Role: {user.role}</span>
        </div>
      </section>

      {/* Plan & billing */}
      <section className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <CreditCard className="h-5 w-5 text-slate-400" />
            Plan & Billing
          </h2>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${planBadge}`}>
            {PLAN_LABELS[user.plan] || user.plan}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-400">Status</span>
            <div className="text-white capitalize">{user.plan_status}</div>
          </div>
          <div>
            <span className="text-slate-400">ASIN limit</span>
            <div className="text-white">{user.asin_limit}</div>
          </div>
          {user.trial_ends_at && (
            <div className="col-span-2">
              <span className="text-slate-400">Trial ends</span>
              <div className="text-white">{new Date(user.trial_ends_at).toLocaleDateString()}</div>
            </div>
          )}
        </div>

        <div className="flex gap-3 pt-2">
          {(user.plan === 'trial' || user.plan === 'starter') && (
            <button
              onClick={() => startCheckout('growth')}
              disabled={checkoutLoading}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 rounded-lg text-white text-sm font-medium transition-colors flex items-center gap-1.5"
            >
              {checkoutLoading ? 'Loading...' : 'Upgrade to Growth'}
              {!checkoutLoading && <ExternalLink className="h-3.5 w-3.5" />}
            </button>
          )}
          <button
            onClick={openBillingPortal}
            disabled={portalLoading}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-white text-sm transition-colors"
          >
            {portalLoading ? 'Loading...' : 'Manage billing'}
          </button>
        </div>
      </section>

      {/* Logout */}
      <button
        onClick={logout}
        className="flex items-center gap-2 text-sm text-red-400 hover:text-red-300 transition-colors"
      >
        <LogOut className="h-4 w-4" />
        Sign out
      </button>
    </div>
  )
}
