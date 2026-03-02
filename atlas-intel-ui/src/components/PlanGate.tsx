import { Link } from 'react-router-dom'
import { Lock } from 'lucide-react'
import { useAuth } from '../auth/AuthContext'

const PLAN_ORDER = ['trial', 'starter', 'growth', 'pro']

interface PlanGateProps {
  minPlan: string
  children: React.ReactNode
  fallback?: React.ReactNode
}

export default function PlanGate({ minPlan, children, fallback }: PlanGateProps) {
  const { user } = useAuth()
  const userIdx = user ? PLAN_ORDER.indexOf(user.plan) : -1
  const minIdx = PLAN_ORDER.indexOf(minPlan)

  if (userIdx >= minIdx) return <>{children}</>

  if (fallback) return <>{fallback}</>

  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="h-14 w-14 rounded-full bg-slate-800 flex items-center justify-center mb-4">
        <Lock className="h-6 w-6 text-slate-500" />
      </div>
      <h2 className="text-lg font-semibold text-white mb-2">
        {minPlan.charAt(0).toUpperCase() + minPlan.slice(1)} plan required
      </h2>
      <p className="text-sm text-slate-400 mb-6 max-w-sm">
        This feature is available on the {minPlan} plan and above.
        Upgrade to unlock advanced competitive intelligence.
      </p>
      <Link
        to="/account"
        className="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white text-sm font-medium transition-colors"
      >
        View plans
      </Link>
    </div>
  )
}
