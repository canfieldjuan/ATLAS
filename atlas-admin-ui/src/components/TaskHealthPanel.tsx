import { useState, useMemo } from 'react'
import { AlertCircle, CircleCheck, CircleX } from 'lucide-react'
import type { TaskHealth } from '../types'
import { isFailing, fmtDuration, fmtSchedule, timeAgo } from '../utils'

const TASK_CATEGORIES: Record<string, { label: string; match: (n: string) => boolean }> = {
  failing: { label: 'Failing', match: () => false },
  b2b: { label: 'B2B Intelligence', match: n => n.startsWith('b2b_') || n === 'challenger_target_discovery' || n === 'vendor_target_enrichment' || n === 'crm_event_processing' },
  consumer: { label: 'Consumer Intelligence', match: n => n.startsWith('complaint_') || n.startsWith('consumer_') || n.startsWith('product_') || n.startsWith('deep_enrichment') || n.startsWith('subcategory_') || n === 'competitive_intelligence' },
  campaigns: { label: 'Campaigns & Outreach', match: n => n.startsWith('campaign_') || n.startsWith('amazon_seller_') || n.startsWith('prospect_') || n === 'trial_nurture' },
  email: { label: 'Email', match: n => n.startsWith('email_') || n === 'gmail_digest' },
  content: { label: 'Content & Blog', match: n => n.includes('blog_post') || n === 'article_enrichment' },
  billing: { label: 'Billing', match: n => n.startsWith('invoice_') || n === 'monthly_invoice_generation' },
  intel: { label: 'Market & News', match: n => n.startsWith('market_') || n.startsWith('news_') || n === 'daily_intelligence' || n === 'weather_traffic_alerts' },
  system: { label: 'System & Scheduling', match: n => ['cleanup_old_executions', 'model_swap_day', 'model_swap_night', 'nightly_memory_sync', 'pattern_learning', 'preference_learning', 'anomaly_detection', 'reasoning_tick', 'reasoning_reflection', 'email_graph_sync'].includes(n) },
  home: { label: 'Home & Personal', match: n => ['calendar_reminder', 'morning_briefing', 'action_escalation', 'proactive_actions', 'device_health_check', 'security_summary', 'departure_auto_fix', 'departure_check'].includes(n) },
}

function taskStatusBorder(task: TaskHealth): string {
  if (!task.enabled) return 'border-l-slate-600'
  if (task.last_status === 'failed' || task.last_status === 'timeout' || task.last_status === 'error')
    return 'border-l-red-400'
  if (task.last_status === 'completed') return 'border-l-emerald-400'
  return 'border-l-amber-400'
}

function taskStatusIcon(task: TaskHealth) {
  if (!task.enabled) return <CircleX className="h-3.5 w-3.5 text-slate-600" />
  if (task.last_status === 'failed' || task.last_status === 'timeout' || task.last_status === 'error')
    return <CircleX className="h-3.5 w-3.5 text-red-400" />
  if (task.last_status === 'completed') return <CircleCheck className="h-3.5 w-3.5 text-emerald-400" />
  return <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
}

export default function TaskHealthPanel({ tasks }: { tasks: TaskHealth[] }) {
  const [expandedCats, setExpandedCats] = useState<Record<string, boolean>>({ failing: true })

  const toggleCat = (key: string) =>
    setExpandedCats(prev => ({ ...prev, [key]: !prev[key] }))

  const tasksByCategory = useMemo(() => {
    const cats: Record<string, TaskHealth[]> = {}
    const failingTasks: TaskHealth[] = []

    for (const t of tasks) {
      if (isFailing(t)) failingTasks.push(t)
      let placed = false
      for (const [key, cat] of Object.entries(TASK_CATEGORIES)) {
        if (key === 'failing') continue
        if (cat.match(t.name)) {
          ;(cats[key] ??= []).push(t)
          placed = true
          break
        }
      }
      if (!placed) (cats['system'] ??= []).push(t)
    }

    for (const arr of Object.values(cats)) {
      arr.sort((a, b) => {
        const aE = isFailing(a), bE = isFailing(b)
        if (aE && !bE) return -1
        if (!aE && bE) return 1
        return a.name.localeCompare(b.name)
      })
    }
    failingTasks.sort((a, b) => a.name.localeCompare(b.name))

    return { cats, failingTasks }
  }, [tasks])

  return (
    <div className="animate-enter mb-8" style={{ animationDelay: '380ms' }}>
      {/* Summary bar */}
      <div className="mb-4 flex items-center gap-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
          Task Health
        </h2>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
            <span className="font-mono text-slate-400">{tasks.filter(t => t.enabled && t.last_status === 'completed').length}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-red-400" />
            <span className="font-mono text-slate-400">{tasksByCategory.failingTasks.length}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-slate-600" />
            <span className="font-mono text-slate-400">{tasks.filter(t => !t.enabled).length}</span>
          </span>
          <span className="text-slate-600">/ {tasks.length} total</span>
        </div>
      </div>

      {/* Failing tasks */}
      {tasksByCategory.failingTasks.length > 0 && (
        <div className="mb-3 rounded-xl border border-red-500/20 bg-red-950/10 p-4">
          <div className="mb-3 flex items-center gap-2">
            <AlertCircle className="h-3.5 w-3.5 text-red-400" />
            <span className="text-[11px] font-semibold uppercase tracking-widest text-red-400">
              Needs Attention ({tasksByCategory.failingTasks.length})
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {tasksByCategory.failingTasks.map(task => (
              <div
                key={`fail-${task.id}`}
                className="rounded-lg border border-red-500/15 border-l-[3px] border-l-red-400 bg-slate-900/60 px-3 py-2.5"
                title={task.last_error || undefined}
              >
                <div className="mb-1.5 flex items-center gap-2">
                  <CircleX className="h-3 w-3 shrink-0 text-red-400" />
                  <span className="truncate font-mono text-[11px] font-medium text-slate-300">{task.name}</span>
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-500">
                  <span>{task.last_run_at ? timeAgo(task.last_run_at) : 'never'}</span>
                  <span className="text-red-400/80">{task.last_status}</span>
                  {task.recent_runs > 0 && (
                    <span className="text-red-400/80">{(task.recent_failure_rate * 100).toFixed(0)}% fail ({task.recent_runs})</span>
                  )}
                </div>
                {task.last_error && (
                  <p className="mt-1.5 truncate text-[10px] text-red-400/60">{task.last_error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Category accordion */}
      <div className="space-y-2">
        {Object.entries(TASK_CATEGORIES).map(([key, cat]) => {
          if (key === 'failing') return null
          const catTasks = tasksByCategory.cats[key]
          if (!catTasks?.length) return null
          const expanded = expandedCats[key] ?? false
          const failCount = catTasks.filter(isFailing).length
          const okCount = catTasks.filter(t => t.enabled && t.last_status === 'completed').length
          return (
            <div key={key} className="rounded-xl border border-slate-800/80 bg-slate-900/40">
              <button
                onClick={() => toggleCat(key)}
                className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-slate-800/20"
              >
                <div className="flex items-center gap-3">
                  <svg
                    className={`h-3 w-3 text-slate-600 transition-transform ${expanded ? 'rotate-90' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                  <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">
                    {cat.label}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  {failCount > 0 && (
                    <span className="rounded-full bg-red-500/15 px-2 py-0.5 font-mono text-red-400">{failCount} failing</span>
                  )}
                  <span className="font-mono text-slate-600">{okCount}/{catTasks.length} ok</span>
                </div>
              </button>
              {expanded && (
                <div className="border-t border-slate-800/60 px-4 py-3">
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {catTasks.map(task => (
                      <div
                        key={task.id}
                        className={`rounded-lg border border-slate-800/60 border-l-[3px] bg-slate-950/40 px-3 py-2.5 transition-colors hover:bg-slate-800/20 ${taskStatusBorder(task)}`}
                        title={task.last_error || undefined}
                      >
                        <div className="mb-1.5 flex items-center gap-2">
                          {taskStatusIcon(task)}
                          <span className="truncate font-mono text-[11px] font-medium text-slate-300">{task.name}</span>
                        </div>
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-500">
                          <span>{task.last_run_at ? timeAgo(task.last_run_at) : 'never'}</span>
                          {task.last_duration_ms != null && <span>{fmtDuration(task.last_duration_ms)}</span>}
                          <span className={
                            task.recent_failure_rate > 0.2 ? 'text-red-400'
                              : task.recent_failure_rate > 0.05 ? 'text-amber-400'
                              : 'text-slate-600'
                          }>
                            {(task.recent_failure_rate * 100).toFixed(0)}%
                          </span>
                          <span className="text-slate-600">{fmtSchedule(task)}</span>
                        </div>
                        {task.last_error && (
                          <p className="mt-1.5 truncate text-[10px] text-red-400/60">{task.last_error}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
