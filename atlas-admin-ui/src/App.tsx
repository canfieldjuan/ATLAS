import { useEffect, useState, useCallback, useMemo } from 'react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, Line,
} from 'recharts'
import {
  DollarSign, Activity, Cpu, Zap, TrendingUp, RefreshCw, AlertCircle, Clock,
  Server, Thermometer, MemoryStick, Wifi, CircleCheck, CircleX,
} from 'lucide-react'

/* ── Types ────────────────────────────────────────────────────── */

interface Summary {
  total_cost_usd: number
  today_cost_usd: number
  total_calls: number
  today_calls: number
  avg_tokens_per_second: number
  total_tokens: number
  total_input_tokens: number
  total_output_tokens: number
  avg_duration_ms: number
}

interface ProviderEntry {
  provider: string
  cost_usd: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  calls: number
}

interface WorkflowEntry {
  workflow: string
  cost_usd: number
  total_tokens: number
  calls: number
  avg_duration_ms: number
}

interface DailyEntry {
  date: string
  cost_usd: number
  total_tokens: number
  calls: number
}

interface RecentCall {
  span_name: string
  model: string
  provider: string
  input_tokens: number
  output_tokens: number
  cost_usd: number
  duration_ms: number
  tokens_per_second: number
  status: string
  created_at: string
}

interface TaskHealth {
  id: string
  name: string
  task_type: string
  schedule_type: string
  cron_expression: string | null
  interval_seconds: number | null
  enabled: boolean
  last_run_at: string | null
  next_run_at: string | null
  last_status: string | null
  last_duration_ms: number | null
  last_error: string | null
  recent_failure_rate: number
  recent_runs: number
}

interface ErrorTimelineEntry {
  date: string
  total_calls: number
  error_calls: number
  error_rate: number
}

interface SystemResources {
  cpu_percent: number
  mem_percent: number
  mem_used_gb: number
  mem_total_gb: number
  net_mbps: number
  gpu: {
    name: string
    utilization_percent: number
    vram_used_gb: number
    vram_total_gb: number
    vram_percent: number
    temperature_c: number
  } | null
}

/* ── Constants ────────────────────────────────────────────────── */

const PROVIDER_COLORS: Record<string, string> = {
  anthropic: '#f97316',
  groq: '#10b981',
  openrouter: '#8b5cf6',
  ollama: '#06b6d4',
  vllm: '#3b82f6',
  deepseek: '#eab308',
  local: '#6b7280',
  unknown: '#475569',
}

/* ── Formatters ───────────────────────────────────────────────── */

function fmtCost(v: number): string {
  if (v >= 100) return `$${v.toFixed(0)}`
  if (v >= 1) return `$${v.toFixed(2)}`
  if (v >= 0.01) return `$${v.toFixed(3)}`
  return `$${v.toFixed(4)}`
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function fmtDuration(ms: number): string {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

function timeAgo(iso: string): string {
  const ts = new Date(iso).getTime()
  if (Number.isNaN(ts)) return ''
  const diff = Date.now() - ts
  if (diff < 0) return 'just now'
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function provColor(p: string): string {
  return PROVIDER_COLORS[p?.toLowerCase()] || PROVIDER_COLORS.unknown
}

function fmtDate(v: string): string {
  return new Date(v + 'T00:00').toLocaleDateString('en', { month: 'short', day: 'numeric' })
}

function gaugeColor(pct: number, thresholds: [number, number] = [50, 80]): string {
  if (pct < thresholds[0]) return 'bg-emerald-400'
  if (pct < thresholds[1]) return 'bg-amber-400'
  return 'bg-red-400'
}

function gaugeTextColor(pct: number, thresholds: [number, number] = [50, 80]): string {
  if (pct < thresholds[0]) return 'text-emerald-400'
  if (pct < thresholds[1]) return 'text-amber-400'
  return 'text-red-400'
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

function fmtSchedule(task: TaskHealth): string {
  if (task.cron_expression) return task.cron_expression
  if (task.interval_seconds) {
    const s = task.interval_seconds
    if (s >= 3600) return `every ${(s / 3600).toFixed(0)}h`
    if (s >= 60) return `every ${(s / 60).toFixed(0)}m`
    return `every ${s}s`
  }
  return task.schedule_type
}

/* ── Stat Card ────────────────────────────────────────────────── */

const CARD_THEMES = {
  cyan: {
    bg: 'from-cyan-500/10 to-cyan-500/5',
    border: 'border-cyan-500/20 hover:border-cyan-500/40',
    icon: 'text-cyan-400',
    glow: 'hover:shadow-cyan-500/5',
  },
  emerald: {
    bg: 'from-emerald-500/10 to-emerald-500/5',
    border: 'border-emerald-500/20 hover:border-emerald-500/40',
    icon: 'text-emerald-400',
    glow: 'hover:shadow-emerald-500/5',
  },
  orange: {
    bg: 'from-orange-500/10 to-orange-500/5',
    border: 'border-orange-500/20 hover:border-orange-500/40',
    icon: 'text-orange-400',
    glow: 'hover:shadow-orange-500/5',
  },
  violet: {
    bg: 'from-violet-500/10 to-violet-500/5',
    border: 'border-violet-500/20 hover:border-violet-500/40',
    icon: 'text-violet-400',
    glow: 'hover:shadow-violet-500/5',
  },
} as const

function StatCard({ icon: Icon, label, value, sub, color = 'cyan', idx = 0 }: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  sub?: string
  color?: keyof typeof CARD_THEMES
  idx?: number
}) {
  const t = CARD_THEMES[color]
  return (
    <div
      className={`animate-enter relative overflow-hidden rounded-xl border bg-gradient-to-br p-5 transition-all duration-300 hover:scale-[1.02] hover:shadow-lg ${t.bg} ${t.border} ${t.glow}`}
      style={{ animationDelay: `${idx * 60}ms` }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            {label}
          </p>
          <p className="mt-2 font-mono text-2xl font-bold tracking-tight text-slate-100">
            {value}
          </p>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        <div className="rounded-lg bg-slate-800/60 p-2.5">
          <Icon className={`h-5 w-5 ${t.icon}`} />
        </div>
      </div>
    </div>
  )
}

/* ── Chart Tooltip ────────────────────────────────────────────── */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CostTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-slate-700/80 bg-slate-800/95 px-3.5 py-2.5 shadow-xl backdrop-blur-sm">
      <p className="mb-1 text-[11px] font-medium text-slate-400">{label}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.filter((p: any) => p.name !== 'error_line').map((p: any, i: number) => (
        <p key={i} className={`font-mono text-sm ${p.name === 'errors' ? 'text-red-400' : 'text-slate-200'}`}>
          {p.name === 'cost' ? fmtCost(p.value)
            : p.name === 'calls' ? `${p.value} calls`
            : p.name === 'errors' ? `${p.value} errors`
            : fmtTokens(p.value)}
        </p>
      ))}
    </div>
  )
}

/* ── Status Dot ───────────────────────────────────────────────── */

function StatusDot({ status }: { status: string }) {
  const color = status === 'error' ? 'bg-red-400' : status === 'completed' ? 'bg-emerald-400' : 'bg-slate-500'
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${color}`} />
}

/* ── App ──────────────────────────────────────────────────────── */

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [providers, setProviders] = useState<ProviderEntry[]>([])
  const [workflows, setWorkflows] = useState<WorkflowEntry[]>([])
  const [daily, setDaily] = useState<DailyEntry[]>([])
  const [recent, setRecent] = useState<RecentCall[]>([])
  const [tasks, setTasks] = useState<TaskHealth[]>([])
  const [errorTimeline, setErrorTimeline] = useState<ErrorTimelineEntry[]>([])
  const [sysResources, setSysResources] = useState<SystemResources | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())
  const [days, setDays] = useState(30)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [sumR, provR, wfR, dailyR, recentR, taskR, errR, sysR] = await Promise.all([
        fetch(`/api/v1/admin/costs/summary?days=${days}`),
        fetch(`/api/v1/admin/costs/by-provider?days=${days}`),
        fetch(`/api/v1/admin/costs/by-workflow?days=${days}`),
        fetch(`/api/v1/admin/costs/daily?days=${days}`),
        fetch('/api/v1/admin/costs/recent?limit=50'),
        fetch(`/api/v1/admin/costs/task-health?days=${days}`),
        fetch(`/api/v1/admin/costs/error-timeline?days=${days}`),
        fetch('/api/v1/admin/costs/system-resources'),
      ])
      const failed = [sumR, provR, wfR, dailyR, recentR].find(r => !r.ok)
      if (failed) throw new Error(`API ${failed.status}: ${failed.statusText}`)
      const [sumD, provD, wfD, dailyD, recentD] = await Promise.all([
        sumR.json(), provR.json(), wfR.json(), dailyR.json(), recentR.json(),
      ])
      setSummary(sumD)
      setProviders((provD.providers || []).map((p: ProviderEntry) => ({ ...p, total_tokens: p.input_tokens + p.output_tokens })))
      setWorkflows(wfD.workflows || [])
      setDaily(dailyD.daily || [])
      setRecent(recentD.calls || [])
      if (taskR.ok) setTasks((await taskR.json()).tasks || [])
      if (errR.ok) setErrorTimeline((await errR.json()).daily || [])
      if (sysR.ok) setSysResources(await sysR.json())
      setLastRefresh(new Date())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }, [days])

  useEffect(() => { fetchAll() }, [fetchAll])

  // Auto-refresh every 60s
  useEffect(() => {
    const id = setInterval(fetchAll, 60_000)
    return () => clearInterval(id)
  }, [fetchAll])

  // System resources fast refresh every 10s
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/v1/admin/costs/system-resources')
        if (r.ok) setSysResources(await r.json())
      } catch { /* ignore */ }
    }, 10_000)
    return () => clearInterval(id)
  }, [])

  // Merge daily + error timeline for chart overlay
  const mergedDaily = useMemo(() => {
    const errMap = new Map(errorTimeline.map(e => [e.date, e]))
    return daily.map(d => ({
      ...d,
      error_calls: errMap.get(d.date)?.error_calls ?? 0,
      error_rate: errMap.get(d.date)?.error_rate ?? 0,
    }))
  }, [daily, errorTimeline])

  // Categorize tasks by domain
  const TASK_CATEGORIES: Record<string, { label: string; match: (n: string) => boolean }> = {
    failing: { label: 'Failing', match: () => false }, // special: populated below
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

  const isFailing = (t: TaskHealth) =>
    t.enabled && (t.last_status === 'failed' || t.last_status === 'timeout' || t.last_status === 'error')

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

    // Sort each category: errors first, then alphabetical
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

  const [expandedCats, setExpandedCats] = useState<Record<string, boolean>>({ failing: true })

  const toggleCat = (key: string) =>
    setExpandedCats(prev => ({ ...prev, [key]: !prev[key] }))

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-200">
      {/* Grid texture */}
      <div className="pointer-events-none fixed inset-0 bg-[linear-gradient(rgba(148,163,184,0.025)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.025)_1px,transparent_1px)] bg-[size:72px_72px]" />

      <div className="relative mx-auto max-w-[1480px] px-6 py-8">
        {/* ── Header ─────────────────────────────── */}
        <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10 ring-1 ring-cyan-500/25">
                <Cpu className="h-5 w-5 text-cyan-400" />
              </div>
              <h1 className="font-display text-2xl font-bold tracking-tight text-slate-100">
                Atlas Cost Monitor
              </h1>
            </div>
            <p className="mt-1.5 pl-[52px] text-[13px] text-slate-500">
              LLM usage analytics &middot; updated {lastRefresh.toLocaleTimeString()}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="rounded-lg border border-slate-700/80 bg-slate-900 px-3 py-2 text-sm text-slate-300 outline-none transition-colors focus:border-cyan-500/50"
            >
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
            </select>
            <button
              onClick={fetchAll}
              disabled={loading}
              className="flex items-center gap-2 rounded-lg border border-slate-700/80 bg-slate-900 px-4 py-2 text-sm text-slate-300 transition-all hover:border-cyan-500/40 hover:text-cyan-400 disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="mb-6 flex items-center gap-3 rounded-lg border border-red-500/30 bg-red-950/40 px-4 py-3 text-sm text-red-400">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {/* ── Stat Cards ─────────────────────────── */}
        <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            icon={DollarSign}
            label="Total Spend"
            value={summary ? fmtCost(summary.total_cost_usd) : '--'}
            sub={summary ? `${days}-day period` : undefined}
            color="cyan"
            idx={0}
          />
          <StatCard
            icon={TrendingUp}
            label="Today's Cost"
            value={summary ? fmtCost(summary.today_cost_usd) : '--'}
            sub={summary ? `${summary.today_calls} calls today` : undefined}
            color="emerald"
            idx={1}
          />
          <StatCard
            icon={Activity}
            label="Total Calls"
            value={summary ? summary.total_calls.toLocaleString() : '--'}
            sub={summary ? `${fmtTokens(summary.total_tokens)} tokens` : undefined}
            color="orange"
            idx={2}
          />
          <StatCard
            icon={Zap}
            label="Avg Throughput"
            value={summary ? `${(summary.avg_tokens_per_second ?? 0).toFixed(1)} t/s` : '--'}
            sub={summary ? `${fmtDuration(summary.avg_duration_ms ?? 0)} avg latency` : undefined}
            color="violet"
            idx={3}
          />
        </div>

        {/* ── System Resources ──────────────────── */}
        {sysResources && (
          <div className="animate-enter mb-8 rounded-xl border border-slate-800/80 bg-slate-900/40 p-4" style={{ animationDelay: '200ms' }}>
            <div className="flex flex-wrap items-center gap-6">
              {/* CPU */}
              <div className="flex min-w-[140px] flex-1 items-center gap-3">
                <Cpu className={`h-4 w-4 shrink-0 ${gaugeTextColor(sysResources.cpu_percent)}`} />
                <div className="flex-1">
                  <div className="mb-1 flex items-baseline justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">CPU</span>
                    <span className={`font-mono text-xs font-medium ${gaugeTextColor(sysResources.cpu_percent)}`}>
                      {sysResources.cpu_percent.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${gaugeColor(sysResources.cpu_percent)}`}
                      style={{ width: `${Math.min(sysResources.cpu_percent, 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* RAM */}
              <div className="flex min-w-[180px] flex-1 items-center gap-3">
                <MemoryStick className={`h-4 w-4 shrink-0 ${gaugeTextColor(sysResources.mem_percent)}`} />
                <div className="flex-1">
                  <div className="mb-1 flex items-baseline justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">RAM</span>
                    <span className={`font-mono text-xs font-medium ${gaugeTextColor(sysResources.mem_percent)}`}>
                      {sysResources.mem_used_gb.toFixed(1)}/{sysResources.mem_total_gb.toFixed(0)} GB
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${gaugeColor(sysResources.mem_percent)}`}
                      style={{ width: `${Math.min(sysResources.mem_percent, 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* GPU VRAM */}
              {sysResources.gpu && (
                <div className="flex min-w-[200px] flex-1 items-center gap-3">
                  <Server className={`h-4 w-4 shrink-0 ${gaugeTextColor(sysResources.gpu.vram_percent)}`} />
                  <div className="flex-1">
                    <div className="mb-1 flex items-baseline justify-between">
                      <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500" title={sysResources.gpu.name}>
                        VRAM
                      </span>
                      <span className={`font-mono text-xs font-medium ${gaugeTextColor(sysResources.gpu.vram_percent)}`}>
                        {sysResources.gpu.vram_used_gb.toFixed(1)}/{sysResources.gpu.vram_total_gb.toFixed(0)} GB
                      </span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${gaugeColor(sysResources.gpu.vram_percent)}`}
                        style={{ width: `${Math.min(sysResources.gpu.vram_percent, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* GPU Temp */}
              {sysResources.gpu && (
                <div className="flex min-w-[100px] items-center gap-3">
                  <Thermometer className={`h-4 w-4 shrink-0 ${gaugeTextColor(sysResources.gpu.temperature_c, [60, 80])}`} />
                  <div>
                    <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">Temp</span>
                    <p className={`font-mono text-xs font-medium ${gaugeTextColor(sysResources.gpu.temperature_c, [60, 80])}`}>
                      {sysResources.gpu.temperature_c}°C
                    </p>
                  </div>
                </div>
              )}

              {/* Network */}
              <div className="flex min-w-[100px] items-center gap-3">
                <Wifi className="h-4 w-4 shrink-0 text-cyan-400" />
                <div>
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">Net</span>
                  <p className="font-mono text-xs font-medium text-cyan-400">
                    {sysResources.net_mbps.toFixed(1)} Mb/s
                  </p>
                </div>
              </div>

              {/* GPU Name */}
              {sysResources.gpu && (
                <div className="hidden text-[10px] text-slate-600 xl:block">
                  {sysResources.gpu.name}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Charts ─────────────────────────────── */}
        <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Daily usage area chart */}
          <div className="animate-enter col-span-1 rounded-xl border border-slate-800/80 bg-slate-900/40 p-5 lg:col-span-2" style={{ animationDelay: '260ms' }}>
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
              Daily Usage
            </h2>
            <div className="h-[300px]">
              {mergedDaily.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-600">
                  No data for this period
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mergedDaily} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.25} />
                        <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="callsGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'DM Sans' }}
                      tickFormatter={fmtDate}
                      stroke="#334155"
                      tickLine={false}
                    />
                    <YAxis
                      yAxisId="tokens"
                      tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                      tickFormatter={(v) => fmtTokens(v)}
                      stroke="#334155"
                      width={48}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      yAxisId="calls"
                      orientation="right"
                      tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                      stroke="#334155"
                      width={32}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip content={<CostTooltip />} />
                    <Area
                      yAxisId="tokens"
                      type="monotone"
                      dataKey="total_tokens"
                      name="tokens"
                      stroke="#22d3ee"
                      strokeWidth={2}
                      fill="url(#costGrad)"
                      dot={false}
                      activeDot={{ r: 4, fill: '#22d3ee', stroke: '#0f172a', strokeWidth: 2 }}
                    />
                    <Area
                      yAxisId="calls"
                      type="monotone"
                      dataKey="calls"
                      name="calls"
                      stroke="#8b5cf6"
                      strokeWidth={1.5}
                      fill="url(#callsGrad)"
                      dot={false}
                      activeDot={{ r: 3, fill: '#8b5cf6', stroke: '#0f172a', strokeWidth: 2 }}
                    />
                    <Area
                      yAxisId="calls"
                      type="monotone"
                      dataKey="error_calls"
                      name="errors"
                      stroke="#f87171"
                      strokeWidth={1.5}
                      fill="#f8717120"
                      dot={false}
                      activeDot={{ r: 3, fill: '#f87171', stroke: '#0f172a', strokeWidth: 2 }}
                    />
                    <Line
                      yAxisId="calls"
                      type="monotone"
                      dataKey="error_calls"
                      name="error_line"
                      stroke="#f87171"
                      strokeWidth={1}
                      strokeDasharray="4 4"
                      dot={false}
                      activeDot={false}
                      legendType="none"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Provider breakdown */}
          <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '320ms' }}>
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
              By Provider
            </h2>
            <div className="h-[300px]">
              {providers.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-600">
                  No data
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={providers} layout="vertical" margin={{ left: 0, right: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                      tickFormatter={fmtTokens}
                      stroke="#334155"
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="provider"
                      tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: 'DM Sans' }}
                      stroke="#334155"
                      width={80}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0].payload as ProviderEntry;
                        return (
                          <div className="rounded-lg border border-slate-700 bg-slate-800/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
                            <p className="mb-1 font-semibold text-slate-200">{d.provider}</p>
                            <p className="text-cyan-400">{fmtTokens(d.total_tokens)} tokens</p>
                            <p className="text-violet-400">{d.calls.toLocaleString()} calls</p>
                            {d.cost_usd > 0 && <p className="text-emerald-400">${d.cost_usd.toFixed(4)}</p>}
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="total_tokens" name="tokens" radius={[0, 4, 4, 0]} barSize={22}>
                      {providers.map((entry, i) => (
                        <Cell key={i} fill={provColor(entry.provider)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* ── Task Health ─────────────────────────── */}
        {tasks.length > 0 && (
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

            {/* Failing tasks — always visible if any */}
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
        )}

        {/* ── Bottom: Workflows + Recent ─────────── */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Workflow table */}
          <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '440ms' }}>
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
              Workflow Breakdown
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-800 text-[10px] uppercase tracking-widest text-slate-600">
                    <th className="pb-3 pr-4 font-semibold">Workflow</th>
                    <th className="pb-3 pr-4 text-right font-semibold">Cost</th>
                    <th className="pb-3 pr-4 text-right font-semibold">Calls</th>
                    <th className="pb-3 pr-4 text-right font-semibold">Tokens</th>
                    <th className="pb-3 text-right font-semibold">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {workflows.length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-12 text-center text-sm text-slate-600">
                        No data yet
                      </td>
                    </tr>
                  )}
                  {workflows.map((w, i) => (
                    <tr
                      key={i}
                      className="border-b border-slate-800/40 transition-colors hover:bg-slate-800/20"
                    >
                      <td className="py-3 pr-4">
                        <span className="font-mono text-xs text-slate-300">{w.workflow}</span>
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-sm text-cyan-400">
                        {fmtCost(w.cost_usd)}
                      </td>
                      <td className="py-3 pr-4 text-right text-slate-400">{w.calls}</td>
                      <td className="py-3 pr-4 text-right text-slate-400">{fmtTokens(w.total_tokens)}</td>
                      <td className="py-3 text-right text-slate-500">
                        <span className="inline-flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {fmtDuration(w.avg_duration_ms)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recent calls feed */}
          <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '500ms' }}>
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
              Recent Calls
            </h2>
            <div className="max-h-[440px] space-y-1 overflow-y-auto pr-1">
              {recent.length === 0 && (
                <p className="py-12 text-center text-sm text-slate-600">No calls recorded yet</p>
              )}
              {recent.map((call, i) => (
                <div
                  key={i}
                  className="group flex items-center gap-3 rounded-lg border border-transparent px-3 py-2.5 transition-colors hover:border-slate-800/80 hover:bg-slate-800/20"
                >
                  <div className="flex flex-col items-center gap-1">
                    <StatusDot status={call.status} />
                    <div
                      className="h-2 w-2 rounded-full opacity-60"
                      style={{ backgroundColor: provColor(call.provider) }}
                    />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span className="truncate font-mono text-xs font-medium text-slate-300">
                        {call.span_name}
                      </span>
                      <span className="shrink-0 text-[10px] text-slate-600">
                        {call.model}
                      </span>
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
                      <span>{(call.input_tokens + call.output_tokens).toLocaleString()} tok</span>
                      <span>{fmtDuration(call.duration_ms)}</span>
                      {call.tokens_per_second > 0 && (
                        <span>{call.tokens_per_second.toFixed(0)} t/s</span>
                      )}
                      <span className="font-mono text-cyan-500/80">{fmtCost(call.cost_usd)}</span>
                    </div>
                  </div>

                  <span className="shrink-0 text-[10px] text-slate-600">
                    {call.created_at ? timeAgo(call.created_at) : ''}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-10 border-t border-slate-800/50 pt-4 text-center text-[11px] text-slate-600">
          Atlas Brain &middot; Cost Monitor &middot; Auto-refreshes every 60s
        </footer>
      </div>
    </div>
  )
}
