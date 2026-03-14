import { useEffect, useState, useCallback, useMemo } from 'react'
import { DollarSign, Activity, Cpu, Zap, TrendingUp, RefreshCw, AlertCircle } from 'lucide-react'
import type {
  Summary, ProviderEntry, WorkflowEntry, DailyEntry, MergedDailyEntry,
  RecentCall, TaskHealth, ErrorTimelineEntry, SystemResources,
  ScrapeSummaryData, ScrapeDetail, ScrapeTopPost,
} from './types'
import { fmtCost, fmtTokens, fmtDuration } from './utils'
import StatCard from './components/StatCard'
import SystemResourcesBar from './components/SystemResourcesBar'
import DailyChart from './components/DailyChart'
import ProviderChart from './components/ProviderChart'
import TaskHealthPanel from './components/TaskHealthPanel'
import ScrapingPipeline from './components/ScrapingPipeline'
import WorkflowTable from './components/WorkflowTable'
import RecentCalls from './components/RecentCalls'

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [providers, setProviders] = useState<ProviderEntry[]>([])
  const [workflows, setWorkflows] = useState<WorkflowEntry[]>([])
  const [daily, setDaily] = useState<DailyEntry[]>([])
  const [recent, setRecent] = useState<RecentCall[]>([])
  const [tasks, setTasks] = useState<TaskHealth[]>([])
  const [errorTimeline, setErrorTimeline] = useState<ErrorTimelineEntry[]>([])
  const [sysResources, setSysResources] = useState<SystemResources | null>(null)
  const [scrapeSummary, setScrapeSummary] = useState<ScrapeSummaryData | null>(null)
  const [scrapeDetails, setScrapeDetails] = useState<ScrapeDetail[]>([])
  const [scrapeTopPosts, setScrapeTopPosts] = useState<ScrapeTopPost[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())
  const [days, setDays] = useState(30)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [sumR, provR, wfR, dailyR, recentR, taskR, errR, sysR, scSumR, scDetR, scTopR] = await Promise.all([
        fetch(`/api/v1/admin/costs/summary?days=${days}`),
        fetch(`/api/v1/admin/costs/by-provider?days=${days}`),
        fetch(`/api/v1/admin/costs/by-workflow?days=${days}`),
        fetch(`/api/v1/admin/costs/daily?days=${days}`),
        fetch('/api/v1/admin/costs/recent?limit=50'),
        fetch(`/api/v1/admin/costs/task-health?days=${days}`),
        fetch(`/api/v1/admin/costs/error-timeline?days=${days}`),
        fetch('/api/v1/admin/costs/system-resources'),
        fetch(`/api/v1/admin/costs/scraping/summary?days=${days}`),
        fetch('/api/v1/admin/costs/scraping/details?limit=50'),
        fetch('/api/v1/admin/costs/scraping/top-posts?source=reddit&limit=25'),
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
      if (scSumR.ok) setScrapeSummary(await scSumR.json())
      if (scDetR.ok) setScrapeDetails((await scDetR.json()).scrapes || [])
      if (scTopR.ok) setScrapeTopPosts((await scTopR.json()).posts || [])
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

  const mergedDaily: MergedDailyEntry[] = useMemo(() => {
    const errMap = new Map(errorTimeline.map(e => [e.date, e]))
    return daily.map(d => ({
      ...d,
      error_calls: errMap.get(d.date)?.error_calls ?? 0,
      error_rate: errMap.get(d.date)?.error_rate ?? 0,
    }))
  }, [daily, errorTimeline])

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
        {sysResources && <SystemResourcesBar data={sysResources} />}

        {/* ── Charts ─────────────────────────────── */}
        <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <DailyChart data={mergedDaily} />
          <ProviderChart data={providers} />
        </div>

        {/* ── Task Health ─────────────────────────── */}
        {tasks.length > 0 && <TaskHealthPanel tasks={tasks} />}

        {/* ── Scraping Pipeline ───────────────────── */}
        <ScrapingPipeline summary={scrapeSummary} details={scrapeDetails} topPosts={scrapeTopPosts} />

        {/* ── Bottom: Workflows + Recent ─────────── */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <WorkflowTable data={workflows} />
          <RecentCalls data={recent} />
        </div>

        {/* Footer */}
        <footer className="mt-10 border-t border-slate-800/50 pt-4 text-center text-[11px] text-slate-600">
          Atlas Brain &middot; Cost Monitor &middot; Auto-refreshes every 60s
        </footer>
      </div>
    </div>
  )
}
