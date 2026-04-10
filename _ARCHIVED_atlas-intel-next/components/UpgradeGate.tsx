import { Lock, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import type { ReactNode } from 'react'

interface UpgradeGateProps {
  allowed: boolean
  feature: string
  requiredPlan: string
  children: ReactNode
}

export default function UpgradeGate({ allowed, feature, requiredPlan, children }: UpgradeGateProps) {
  if (allowed) return <>{children}</>

  return (
    <div className="flex flex-col items-center justify-center py-24 px-6 text-center">
      <div className="w-14 h-14 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-5">
        <Lock className="h-6 w-6 text-slate-400" />
      </div>
      <h2 className="text-xl font-semibold text-white mb-2">
        {feature} requires {requiredPlan}
      </h2>
      <p className="text-sm text-slate-400 max-w-md mb-6">
        Upgrade your plan to unlock {feature.toLowerCase()} and get more from your churn intelligence.
      </p>
      <Link
        href="/account"
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-medium hover:bg-cyan-500/20 transition-colors"
      >
        Upgrade Plan
        <ArrowRight className="h-4 w-4" />
      </Link>
    </div>
  )
}
