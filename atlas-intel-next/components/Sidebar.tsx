"use client";

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  LayoutDashboard,
  Building2,
  MessageSquareText,
  FileBarChart,
  Crosshair,
  Shield,
  Swords,
  Handshake,
  Newspaper,
  FileSearch,
  MailSearch,
  Users,
  LogOut,
  Lock,
  X,
} from 'lucide-react'
import AtlasRobotLogo from '@/components/AtlasRobotLogo'
import { useAuth } from '@/lib/auth/AuthContext'
import { usePlanGate } from '@/lib/hooks/usePlanGate'
import { clsx } from 'clsx'

interface SidebarLink {
  to: string
  icon: typeof LayoutDashboard
  label: string
  gate?: 'campaigns' | 'reports'
}

const links: SidebarLink[] = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/vendors', icon: Building2, label: 'Vendors' },
  { to: '/reviews', icon: MessageSquareText, label: 'Reviews' },
  { to: '/reports', icon: FileBarChart, label: 'Reports', gate: 'reports' },
  { to: '/leads', icon: Crosshair, label: 'Leads' },
  { to: '/vendor-targets', icon: Shield, label: 'Targets' },
  { to: '/challengers', icon: Swords, label: 'Challengers' },
  { to: '/affiliates', icon: Handshake, label: 'Affiliates' },
  { to: '/blog', icon: Newspaper, label: 'Blog' },
]

const auditLinks: SidebarLink[] = [
  { to: '/blog-review', icon: FileSearch, label: 'Blog Review' },
  { to: '/campaign-review', icon: MailSearch, label: 'Campaigns' },
  { to: '/briefing-review', icon: MailSearch, label: 'Briefings' },
  { to: '/prospects', icon: Users, label: 'Prospects' },
]

interface SidebarProps {
  open: boolean
  onClose: () => void
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  const pathname = usePathname()
  const { user, logout } = useAuth()
  const { canAccessCampaigns, canAccessReports } = usePlanGate()

  const gateMap: Record<string, boolean> = {
    campaigns: canAccessCampaigns,
    reports: canAccessReports,
  }

  return (
    <>
      {/* Backdrop (mobile) */}
      {open && (
        <div
          className="fixed inset-0 bg-black/50 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={clsx(
          'fixed left-0 top-0 bottom-0 w-56 bg-slate-900/70 border-r border-slate-700/50 backdrop-blur flex flex-col z-30 transition-transform duration-200',
          open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        )}
      >
        <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AtlasRobotLogo className="h-6 w-6" />
            <span className="text-lg font-semibold text-white">
              Churn Signals
            </span>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden text-slate-400 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {links.map(({ to, icon: Icon, label, gate }) => {
            const locked = gate ? !gateMap[gate] : false
            return (
              <Link
                key={to}
                href={to}
                onClick={onClose}
                className={clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors',
                  (to === '/' ? pathname === to : pathname.startsWith(to))
                    ? 'bg-cyan-500/10 text-cyan-400'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
                {locked && <Lock className="h-3 w-3 ml-auto text-slate-600" />}
              </Link>
            )
          })}

          {/* Audit section */}
          <div className="pt-3 mt-3 border-t border-slate-700/50">
            <span className="px-3 text-[10px] font-semibold uppercase tracking-wider text-slate-600">
              Audit
            </span>
            <div className="mt-1 space-y-1">
              {auditLinks.map(({ to, icon: Icon, label }) => (
                <Link
                  key={to}
                  href={to}
                  onClick={onClose}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors',
                    pathname.startsWith(to)
                      ? 'bg-cyan-500/10 text-cyan-400'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              ))}
            </div>
          </div>
        </nav>
        {user && (
          <div className="p-3 border-t border-slate-700/50">
            <Link href="/account" className="block px-3 py-1 mb-2 text-xs text-slate-500 truncate hover:text-slate-300 transition-colors">{user.email}</Link>
            <button
              onClick={logout}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors w-full"
            >
              <LogOut className="h-4 w-4" />
              Sign out
            </button>
          </div>
        )}
      </aside>
    </>
  )
}
