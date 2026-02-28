import { Activity, Clock, Database, Target } from 'lucide-react'
import type { PipelineStatus as PipelineStatusType } from '../types'

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return 'Never'
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function Skeleton() {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 animate-pulse">
      <div className="flex items-center gap-2 mb-4">
        <div className="h-4 w-4 bg-slate-700/50 rounded" />
        <div className="h-4 w-28 bg-slate-700/50 rounded" />
      </div>
      <div className="space-y-3">
        <div className="flex justify-between">
          <div className="h-3 w-24 bg-slate-700/50 rounded" />
          <div className="h-3 w-8 bg-slate-700/50 rounded" />
        </div>
        <div className="h-2 bg-slate-700/50 rounded-full" />
        <div className="grid grid-cols-2 gap-3">
          <div className="h-3 w-20 bg-slate-700/50 rounded" />
          <div className="h-3 w-20 bg-slate-700/50 rounded" />
        </div>
        <div className="pt-2 border-t border-slate-800 space-y-2">
          <div className="h-3 w-full bg-slate-700/50 rounded" />
          <div className="h-3 w-full bg-slate-700/50 rounded" />
          <div className="h-3 w-3/4 bg-slate-700/50 rounded" />
        </div>
      </div>
    </div>
  )
}

export default function PipelineStatusWidget({ data }: { data: PipelineStatusType | null }) {
  if (!data) {
    return <Skeleton />
  }

  const enriched = data.enrichment_counts['enriched'] ?? 0
  const pending = data.enrichment_counts['pending'] ?? 0
  const total = Object.values(data.enrichment_counts).reduce((a, b) => a + b, 0)
  const rate = total > 0 ? Math.round((enriched / total) * 100) : 0

  return (
    <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
      <h3 className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
        <Activity className="h-4 w-4 text-cyan-400" />
        Pipeline Health
      </h3>
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="flex items-center gap-2 text-slate-400">
            <Database className="h-3.5 w-3.5" />
            Enrichment Rate
          </span>
          <span className="text-white font-medium">{rate}%</span>
        </div>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full transition-all"
            style={{ width: `${rate}%` }}
          />
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="flex items-center gap-1.5 text-slate-400">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            Enriched: {enriched}
          </div>
          <div className="flex items-center gap-1.5 text-slate-400">
            <span className="h-2 w-2 rounded-full bg-amber-400" />
            Pending: {pending}
          </div>
        </div>
        <div className="pt-2 border-t border-slate-800 space-y-2 text-xs text-slate-400">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Clock className="h-3 w-3" />
              Last Enrichment
            </span>
            <span>{timeAgo(data.last_enrichment_at)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Target className="h-3 w-3" />
              Active Targets
            </span>
            <span>{data.active_scrape_targets}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Imports (24h)</span>
            <span>{data.recent_imports_24h}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
