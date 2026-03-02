import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Tag,
  Scale,
  GitCompareArrows,
  Lightbulb,
  ShieldAlert,
  MessageSquareText,
  Search,
  X,
  User,
  Lock,
  Activity,
  Target,
  TrendingDown,
  Users,
  FileText,
  Megaphone,
} from 'lucide-react'
import { clsx } from 'clsx'
import { useAuth } from '../auth/AuthContext'

const PLAN_BADGES: Record<string, string> = {
  trial: 'bg-slate-700 text-slate-300',
  starter: 'bg-cyan-900/50 text-cyan-300',
  growth: 'bg-violet-900/50 text-violet-300',
  pro: 'bg-amber-900/50 text-amber-300',
  b2b_trial: 'bg-slate-700 text-slate-300',
  b2b_starter: 'bg-cyan-900/50 text-cyan-300',
  b2b_growth: 'bg-violet-900/50 text-violet-300',
  b2b_pro: 'bg-amber-900/50 text-amber-300',
}

interface NavItem {
  to: string
  icon: typeof LayoutDashboard
  label: string
  minPlan?: string
}

const PLAN_ORDER = ['trial', 'starter', 'growth', 'pro']
const B2B_PLAN_ORDER = ['b2b_trial', 'b2b_starter', 'b2b_growth', 'b2b_pro']

const consumerLinks: NavItem[] = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/brands', icon: Tag, label: 'Brands' },
  { to: '/compare', icon: Scale, label: 'Compare', minPlan: 'growth' },
  { to: '/flows', icon: GitCompareArrows, label: 'Competitive Flows', minPlan: 'growth' },
  { to: '/features', icon: Lightbulb, label: 'Feature Gaps' },
  { to: '/safety', icon: ShieldAlert, label: 'Safety Signals' },
  { to: '/reviews', icon: MessageSquareText, label: 'Reviews' },
]

const b2bRetentionLinks: NavItem[] = [
  { to: '/b2b', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/b2b/signals', icon: Activity, label: 'Churn Signals' },
  { to: '/b2b/displacement', icon: TrendingDown, label: 'Displacement' },
  { to: '/b2b/reviews', icon: MessageSquareText, label: 'Reviews' },
  { to: '/b2b/reports', icon: FileText, label: 'Reports', minPlan: 'b2b_starter' },
  { to: '/b2b/campaigns', icon: Megaphone, label: 'Campaigns', minPlan: 'b2b_growth' },
]

const b2bChallengerLinks: NavItem[] = [
  { to: '/b2b', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/b2b/leads', icon: Target, label: 'Leads' },
  { to: '/b2b/signals', icon: Activity, label: 'Competitor Gaps' },
  { to: '/b2b/displacement', icon: TrendingDown, label: 'Displacement' },
  { to: '/b2b/campaigns', icon: Megaphone, label: 'Campaigns', minPlan: 'b2b_growth' },
  { to: '/b2b/reports', icon: FileText, label: 'Reports', minPlan: 'b2b_starter' },
  { to: '/b2b/reviews', icon: MessageSquareText, label: 'Reviews' },
]

const PRODUCT_TITLES: Record<string, string> = {
  consumer: 'Consumer Intel',
  b2b_retention: 'Vendor Intel',
  b2b_challenger: 'Challenger Intel',
}

interface SidebarProps {
  open: boolean
  onClose: () => void
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  const { user, logout } = useAuth()
  const product = user?.product || 'consumer'
  const isB2B = product === 'b2b_retention' || product === 'b2b_challenger'
  const planOrder = isB2B ? B2B_PLAN_ORDER : PLAN_ORDER
  const userPlanIdx = user ? planOrder.indexOf(user.plan) : -1

  const links = product === 'b2b_challenger'
    ? b2bChallengerLinks
    : product === 'b2b_retention'
      ? b2bRetentionLinks
      : consumerLinks

  const title = PRODUCT_TITLES[product] || 'Consumer Intel'
  const IconComponent = isB2B ? Users : Search

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
            <IconComponent className="h-6 w-6 text-cyan-400" />
            <span className="text-lg font-semibold text-white">
              {title}
            </span>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden text-slate-400 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {links.map(({ to, icon: Icon, label, minPlan }) => {
            const locked = minPlan && userPlanIdx < planOrder.indexOf(minPlan)
            return (
              <NavLink
                key={to}
                to={locked ? '#' : to}
                end={to === '/' || to === '/b2b'}
                onClick={e => { if (locked) e.preventDefault(); else onClose() }}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors',
                    locked
                      ? 'text-slate-600 cursor-not-allowed'
                      : isActive
                        ? 'bg-cyan-500/10 text-cyan-400'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
                {locked && <Lock className="h-3 w-3 ml-auto text-slate-600" />}
              </NavLink>
            )
          })}
        </nav>

        {/* User section */}
        {user && (
          <div className="p-3 border-t border-slate-700/50 space-y-2">
            <NavLink
              to="/account"
              onClick={onClose}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors',
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-400'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                )
              }
            >
              <User className="h-4 w-4" />
              <span className="truncate flex-1">{user.full_name || user.email}</span>
              <span className={clsx('px-1.5 py-0.5 rounded text-[10px] font-medium', PLAN_BADGES[user.plan] || PLAN_BADGES.trial)}>
                {user.plan.replace('b2b_', '').toUpperCase()}
              </span>
            </NavLink>
            <button
              onClick={logout}
              className="w-full text-left px-3 py-2 text-xs text-slate-500 hover:text-red-400 transition-colors"
            >
              Sign out
            </button>
          </div>
        )}
      </aside>
    </>
  )
}
