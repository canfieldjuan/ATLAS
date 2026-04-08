import useApiData from '../hooks/useApiData'
import {
  Activity,
  Send,
  Eye,
  MousePointerClick,
  MessageSquare,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Zap,
  Target,
  Loader2,
  Clock,
} from 'lucide-react'
import { clsx } from 'clsx'
import { fetchCompanyTimeline } from '../api/client'
import type { TimelineEvent } from '../api/client'

interface CompanyTimelineProps {
  company: string
  vendor: string
}

const EVENT_CONFIG: Record<string, { icon: typeof Activity; color: string; label: string }> = {
  generated: { icon: Zap, color: 'text-cyan-400', label: 'Campaign generated' },
  approved: { icon: CheckCircle2, color: 'text-green-400', label: 'Approved' },
  queued: { icon: Send, color: 'text-cyan-400', label: 'Queued for send' },
  sent: { icon: Send, color: 'text-green-400', label: 'Sent' },
  delivered: { icon: Send, color: 'text-green-300', label: 'Delivered' },
  opened: { icon: Eye, color: 'text-amber-400', label: 'Opened' },
  clicked: { icon: MousePointerClick, color: 'text-amber-300', label: 'Clicked' },
  replied: { icon: MessageSquare, color: 'text-green-400', label: 'Replied' },
  bounced: { icon: XCircle, color: 'text-red-400', label: 'Bounced' },
  send_failed: { icon: AlertTriangle, color: 'text-red-400', label: 'Send failed' },
  cancelled: { icon: XCircle, color: 'text-slate-400', label: 'Cancelled' },
  suppressed: { icon: XCircle, color: 'text-slate-500', label: 'Suppressed' },
  generation_failure: { icon: AlertTriangle, color: 'text-red-400', label: 'Generation failed' },
  paused: { icon: Clock, color: 'text-amber-400', label: 'Paused' },
  resumed: { icon: Activity, color: 'text-cyan-400', label: 'Resumed' },
}

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return '--'
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return '--'
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

const OUTCOME_COLORS: Record<string, string> = {
  meeting_booked: 'bg-green-500/20 text-green-400',
  deal_opened: 'bg-cyan-500/20 text-cyan-400',
  deal_won: 'bg-green-500/20 text-green-300',
  deal_lost: 'bg-red-500/20 text-red-400',
  no_opportunity: 'bg-slate-500/20 text-slate-400',
  disqualified: 'bg-slate-500/20 text-slate-500',
  pending: 'bg-slate-700/20 text-slate-500',
}

export default function CompanyTimeline({ company, vendor }: CompanyTimelineProps) {
  const { data, loading, error } = useApiData(
    () => fetchCompanyTimeline({ company, vendor }),
    [company, vendor],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )
  const events: TimelineEvent[] = data?.events ?? []

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8 text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin mr-2" />
        Loading timeline...
      </div>
    )
  }

  if (error) {
    return <div className="text-sm text-red-400 py-4">{error.message}</div>
  }

  if (events.length === 0) {
    return <div className="text-sm text-slate-500 py-4">No campaign activity yet for this company.</div>
  }

  return (
    <div className="space-y-1">
      {/* Sequence summary cards */}
      {events.filter((e) => e.type === 'sequence_state').map((seq, i) => (
        <div key={`seq-${i}`} className="bg-slate-800/40 border border-slate-700/30 rounded-lg p-3 mb-3">
          <div className="flex items-center gap-2 mb-2">
            <Target className="h-4 w-4 text-cyan-400" />
            <span className="text-sm font-medium text-white">
              Sequence {seq.status === 'replied' ? '(replied)' : seq.status === 'completed' ? '(completed)' : `(${seq.status})`}
            </span>
            {seq.recipient && <span className="text-xs text-slate-500">{seq.recipient}</span>}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div>
              <span className="text-slate-500">Step </span>
              <span className="text-white">{seq.step ?? 0}/{seq.max_steps ?? 0}</span>
            </div>
            <div>
              <span className="text-slate-500">Opens </span>
              <span className="text-white">{seq.open_count ?? 0}</span>
            </div>
            <div>
              <span className="text-slate-500">Clicks </span>
              <span className="text-white">{seq.click_count ?? 0}</span>
            </div>
            {seq.outcome && seq.outcome !== 'pending' && (
              <div>
                <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium', OUTCOME_COLORS[seq.outcome] || 'bg-slate-700/20 text-slate-400')}>
                  {seq.outcome.replace(/_/g, ' ')}
                </span>
                {seq.outcome_revenue != null && seq.outcome_revenue > 0 && (
                  <span className="ml-1 text-green-400">${seq.outcome_revenue.toLocaleString()}</span>
                )}
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Signal detection */}
      {events.filter((e) => e.type === 'signal_detected').map((sig, i) => (
        <div key={`sig-${i}`} className="bg-cyan-500/5 border border-cyan-500/20 rounded-lg p-3 mb-3">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-cyan-400" />
            <span className="text-sm font-medium text-cyan-300">Signal detected</span>
            <span className="text-xs text-slate-500">{formatTimestamp(sig.timestamp)}</span>
          </div>
          <div className="mt-1 flex flex-wrap gap-2 text-xs">
            {sig.buying_stage && <span className="px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300">{sig.buying_stage.replace(/_/g, ' ')}</span>}
            {sig.role_type && <span className="px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300">{sig.role_type}</span>}
            {sig.urgency_score != null && <span className="px-1.5 py-0.5 rounded bg-red-500/10 text-red-300">urgency {sig.urgency_score}</span>}
            {sig.seat_count != null && <span className="px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300">{sig.seat_count} seats</span>}
          </div>
        </div>
      ))}

      {/* Campaign event timeline */}
      <div className="relative pl-6 border-l border-slate-700/50">
        {events.filter((e) => e.type === 'campaign_event').map((ev, i) => {
          const cfg = EVENT_CONFIG[ev.event || ''] || { icon: Activity, color: 'text-slate-400', label: ev.event || 'Unknown' }
          const Icon = cfg.icon
          return (
            <div key={i} className="relative mb-3 last:mb-0">
              <div className={clsx('absolute -left-[25px] top-0.5 h-4 w-4 rounded-full border-2 border-slate-800 flex items-center justify-center', cfg.color)}>
                <Icon className="h-2.5 w-2.5" />
              </div>
              <div className="flex items-baseline gap-2">
                <span className={clsx('text-xs font-medium', cfg.color)}>{cfg.label}</span>
                <span className="text-[10px] text-slate-600">{formatTimestamp(ev.timestamp)}</span>
              </div>
              {ev.subject && <p className="text-xs text-slate-400 mt-0.5 truncate max-w-md">{ev.subject}</p>}
              {ev.recipient && <p className="text-[10px] text-slate-600 mt-0.5">{ev.recipient}</p>}
              {ev.error && <p className="text-[10px] text-red-400 mt-0.5">{ev.error}</p>}
            </div>
          )
        })}
      </div>
    </div>
  )
}
