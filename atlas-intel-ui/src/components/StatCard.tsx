import type { ReactNode } from 'react'

interface StatCardProps {
  label: string
  value: string | number
  icon: ReactNode
  sub?: string
  skeleton?: boolean
}

export default function StatCard({ label, value, icon, sub, skeleton }: StatCardProps) {
  if (skeleton) {
    return (
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5 animate-pulse">
        <div className="flex items-center justify-between mb-3">
          <div className="h-4 w-24 bg-slate-700/50 rounded" />
          <div className="h-5 w-5 bg-slate-700/50 rounded" />
        </div>
        <div className="h-7 w-16 bg-slate-700/50 rounded" />
        {sub !== undefined && <div className="h-3 w-20 bg-slate-700/50 rounded mt-2" />}
      </div>
    )
  }

  return (
    <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-slate-400">{label}</span>
        <span className="text-cyan-400">{icon}</span>
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  )
}
