"use client";

import { useState } from 'react'
import { useAuth } from '@/lib/auth/AuthContext'
import { CreditCard, AlertTriangle, LogOut, ExternalLink, AlertCircle } from 'lucide-react'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

const PLAN_LABELS: Record<string, string> = {
  b2b_trial: 'Trial',
  b2b_starter: 'Starter',
  b2b_growth: 'Growth',
  b2b_pro: 'Pro',
  trial: 'Trial',
  starter: 'Starter',
  growth: 'Growth',
  pro: 'Pro',
}

const PLAN_COLORS: Record<string, string> = {
  b2b_trial: 'bg-slate-700 text-slate-300',
  b2b_starter: 'bg-cyan-900/50 text-cyan-300',
  b2b_growth: 'bg-violet-900/50 text-violet-300',
  b2b_pro: 'bg-amber-900/50 text-amber-300',
  trial: 'bg-slate-700 text-slate-300',
  starter: 'bg-cyan-900/50 text-cyan-300',
  growth: 'bg-violet-900/50 text-violet-300',
  pro: 'bg-amber-900/50 text-amber-300',
}

const VENDOR_LIMITS: Record<string, number | string> = {
  b2b_starter: 5,
  b2b_growth: 25,
  b2b_pro: 'Unlimited',
}

export default function Account() {
  const { user, logout } = useAuth()
  const [portalLoading, setPortalLoading] = useState(false)
  const [checkoutLoading, setCheckoutLoading] = useState('')
  const [error, setError] = useState('')

  if (!user) return null

  const isPastDue = user.plan_status === 'past_due'
  const isTrialExpired = user.plan_status === 'canceled' || (
    user.trial_ends_at && new Date(user.trial_ends_at) < new Date()
  )
  const canUpgrade = user.plan === 'b2b_trial' || user.plan === 'b2b_starter' || user.plan === 'trial' || user.plan === 'starter'
  const hasBilling = user.plan_status === 'active' || user.plan_status === 'past_due'
  const planBadge = PLAN_COLORS[user.plan] || PLAN_COLORS.trial

  async function startCheckout(planName: string) {
    setCheckoutLoading(planName)
    setError('')
    try {
      const token = (typeof window !== 'undefined' ? localStorage.getItem('atlas_token') : null)
      const res = await fetch(`${API_BASE}/api/v1/billing/checkout`, {
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
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Failed to start checkout' }))
        throw new Error(body.detail || `Error ${res.status}`)
      }
      const data = await res.json()
      window.location.href = data.checkout_url
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start checkout')
    } finally {
      setCheckoutLoading('')
    }
  }

  async function openBillingPortal() {
    setPortalLoading(true)
    setError('')
    try {
      const token = (typeof window !== 'undefined' ? localStorage.getItem('atlas_token') : null)
      const res = await fetch(`${API_BASE}/api/v1/billing/portal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Failed to open billing portal' }))
        throw new Error(body.detail || `Error ${res.status}`)
      }
      const data = await res.json()
      window.location.href = data.portal_url
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open billing portal')
    } finally {
      setPortalLoading(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto py-12 px-4 space-y-6">
      <h1 className="text-2xl font-bold text-white">Account</h1>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Profile */}
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-cyan-900/30 flex items-center justify-center">
            <span className="text-cyan-400 font-bold text-sm">
              {user.full_name?.[0]?.toUpperCase() ?? user.email?.[0]?.toUpperCase() ?? '?'}
            </span>
          </div>
          <div>
            <div className="text-white font-medium">{user.full_name ?? 'User'}</div>
            <div className="text-sm text-slate-400">{user.email}</div>
          </div>
        </div>

        <div className="border-t border-slate-700/50 pt-4 space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Company</span>
            <span className="text-white">{user.account_name ?? '--'}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Plan</span>
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${planBadge}`}>
              {PLAN_LABELS[user.plan] || user.plan}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Status</span>
            <span className={
              isPastDue ? 'text-red-400 font-medium' :
              isTrialExpired ? 'text-amber-400 font-medium' :
              'text-green-400 font-medium'
            }>
              {isPastDue ? 'Past Due' : isTrialExpired ? 'Trial Expired' : user.plan_status?.replace(/_/g, ' ') ?? '--'}
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Vendor Limit</span>
            <span className="text-white">{user.vendor_limit ?? '--'}</span>
          </div>
          {user.trial_ends_at && (
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Trial Ends</span>
              <span className="text-white">{new Date(user.trial_ends_at).toLocaleDateString()}</span>
            </div>
          )}
        </div>

        {(isPastDue || isTrialExpired) && (
          <div className="flex items-start gap-2 text-sm text-amber-400 bg-amber-900/20 border border-amber-800/50 rounded-lg px-3 py-2">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <div>
              {isPastDue
                ? 'Your payment is past due. Please update your billing information.'
                : 'Your free trial has ended. Upgrade to continue accessing churn intelligence.'}
            </div>
          </div>
        )}
      </div>

      {/* Upgrade plans */}
      {canUpgrade && (
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <CreditCard className="h-5 w-5 text-slate-400" />
            Upgrade Plan
          </h2>
          <div className="grid gap-3">
            {user.plan !== 'b2b_growth' && user.plan !== 'growth' && (
              <button
                onClick={() => startCheckout(user.product?.startsWith('b2b') ? 'b2b_growth' : 'growth')}
                disabled={!!checkoutLoading}
                className="flex items-center justify-between px-4 py-3 bg-violet-600/20 hover:bg-violet-600/30 border border-violet-500/30 rounded-lg transition-colors disabled:opacity-50"
              >
                <div className="text-left">
                  <div className="text-white font-medium">Growth</div>
                  <div className="text-sm text-slate-400">
                    {VENDOR_LIMITS.b2b_growth} vendors, campaigns, reports
                  </div>
                </div>
                <div className="flex items-center gap-1.5 text-violet-300 text-sm font-medium">
                  {checkoutLoading === 'b2b_growth' || checkoutLoading === 'growth' ? 'Loading...' : 'Upgrade'}
                  {!checkoutLoading && <ExternalLink className="h-3.5 w-3.5" />}
                </div>
              </button>
            )}
            <button
              onClick={() => startCheckout(user.product?.startsWith('b2b') ? 'b2b_pro' : 'pro')}
              disabled={!!checkoutLoading}
              className="flex items-center justify-between px-4 py-3 bg-amber-600/20 hover:bg-amber-600/30 border border-amber-500/30 rounded-lg transition-colors disabled:opacity-50"
            >
              <div className="text-left">
                <div className="text-white font-medium">Pro</div>
                <div className="text-sm text-slate-400">
                  {VENDOR_LIMITS.b2b_pro} vendors, campaigns, reports, API access
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-amber-300 text-sm font-medium">
                {checkoutLoading === 'b2b_pro' || checkoutLoading === 'pro' ? 'Loading...' : 'Upgrade'}
                {!checkoutLoading && <ExternalLink className="h-3.5 w-3.5" />}
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Billing management */}
      <div className="flex items-center gap-3">
        {hasBilling && (
          <button
            onClick={openBillingPortal}
            disabled={portalLoading}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-white text-sm transition-colors"
          >
            <CreditCard className="h-4 w-4" />
            {portalLoading ? 'Loading...' : 'Manage billing'}
          </button>
        )}
        <button
          onClick={logout}
          className="inline-flex items-center gap-2 px-4 py-2.5 border border-slate-600 hover:border-slate-500 rounded-lg text-slate-300 text-sm transition-colors"
        >
          <LogOut className="h-4 w-4" />
          Sign out
        </button>
      </div>
    </div>
  )
}
