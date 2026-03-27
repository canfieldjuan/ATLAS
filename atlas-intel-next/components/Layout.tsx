"use client";

import { useState, type ReactNode } from 'react'
import { Menu, Search, AlertTriangle, CreditCard } from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import { useAuth } from '@/lib/auth/AuthContext'

function TrialBanner() {
  const { user } = useAuth()
  const [now] = useState(() => Date.now())
  if (!user || !user.trial_ends_at) return null
  if (!['trial', 'b2b_trial'].includes(user.plan)) return null

  const ends = new Date(user.trial_ends_at).getTime()
  const daysLeft = Math.ceil((ends - now) / (1000 * 60 * 60 * 24))

  if (daysLeft > 7) return null

  const expired = daysLeft <= 0
  const bg = expired
    ? 'bg-red-900/80 border-red-700/50'
    : 'bg-amber-900/80 border-amber-700/50'
  const text = expired
    ? 'Your trial has expired. Upgrade to continue accessing your dashboard.'
    : `Your trial expires in ${daysLeft} day${daysLeft === 1 ? '' : 's'}. Upgrade now to keep your data.`

  return (
    <div className={`${bg} border-b px-4 py-2.5 flex items-center justify-between gap-3`}>
      <div className="flex items-center gap-2 text-sm">
        <AlertTriangle className="h-4 w-4 shrink-0" />
        <span>{text}</span>
      </div>
      <a
        href="/account"
        className="shrink-0 text-xs font-medium px-3 py-1 rounded bg-white/10 hover:bg-white/20 transition"
      >
        Upgrade
      </a>
    </div>
  )
}

function PaymentBanner() {
  const { user } = useAuth()
  if (!user || user.plan_status !== 'past_due') return null

  return (
    <div className="bg-red-900/80 border-b border-red-700/50 px-4 py-2.5 flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 text-sm">
        <CreditCard className="h-4 w-4 shrink-0" />
        <span>Your payment is past due. Update your billing to restore access.</span>
      </div>
      <a
        href="/account"
        className="shrink-0 text-xs font-medium px-3 py-1 rounded bg-white/10 hover:bg-white/20 transition"
      >
        Fix Billing
      </a>
    </div>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen">
      {/* Status banners */}
      <TrialBanner />
      <PaymentBanner />

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
