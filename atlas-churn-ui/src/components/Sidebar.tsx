import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Building2,
  MessageSquareText,
  FileBarChart,
  Activity,
} from 'lucide-react'
import { clsx } from 'clsx'

const links = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/vendors', icon: Building2, label: 'Vendors' },
  { to: '/reviews', icon: MessageSquareText, label: 'Reviews' },
  { to: '/reports', icon: FileBarChart, label: 'Reports' },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 bottom-0 w-56 bg-slate-900/70 border-r border-slate-700/50 backdrop-blur flex flex-col z-30">
      <div className="p-4 border-b border-slate-700/50">
        <div className="flex items-center gap-2">
          <Activity className="h-6 w-6 text-cyan-400" />
          <span className="text-lg font-semibold text-white">
            Churn Intel
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-1">B2B Intelligence</p>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors',
                isActive
                  ? 'bg-cyan-500/10 text-cyan-400'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
