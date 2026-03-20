import { Suspense, lazy, useEffect, useState, useCallback, useMemo } from 'react'
import { DollarSign, Activity, Cpu, Zap, TrendingUp, RefreshCw, AlertCircle } from 'lucide-react'
import type {
  Summary, ProviderEntry, WorkflowEntry, DailyEntry, MergedDailyEntry,
  RecentCall, TaskHealth, ErrorTimelineEntry, SystemResources,
  ScrapeSummaryData, ScrapeDetail, ScrapeTopPost, ModelEntry,
  ReasoningActivityData, RedditOverview, RedditBySubredditData,
  RedditSignalBreakdownData, RedditPerVendorData,
} from './types'
import { fmtCost, fmtTokens, fmtDuration } from './utils'
import { fetchAdminDashboardData, fetchSystemResources } from './api/client'
import StatCard from './components/StatCard'
import SystemResourcesBar from './components/SystemResourcesBar'
import DailyChart from './components/DailyChart'
import ProviderChart from './components/ProviderChart'

const ModelTable = lazy(() => import('./components/ModelTable'))
const TaskHealthPanel = lazy(() => import('./components/TaskHealthPanel'))
const ScrapingPipeline = lazy(() => import('./components/ScrapingPipeline'))
const WorkflowTable = lazy(() => import('./components/WorkflowTable'))
const RecentCalls = lazy(() => import('./components/RecentCalls'))
const ReasoningPanel = lazy(() => import('./components/ReasoningPanel'))

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [providers, setProviders] = useState<ProviderEntry[]>([])
  const [models, setModels] = useState<ModelEntry[]>([])
  const [workflows, setWorkflows] = useState<WorkflowEntry[]>([])
  const [daily, setDaily] = useState<DailyEntry[]>([])
  const [recent, setRecent] = useState<RecentCall[]>([])
  const [reasoningActivity, setReasoningActivity] = useState<ReasoningActivityData | null>(null)
  const [tasks, setTasks] = useState<TaskHealth[]>([])
  const [errorTimeline, setErrorTimeline] = useState<ErrorTimelineEntry[]>([])
  const [sysResources, setSysResources] = useState<SystemResources | null>(null)
  const [scrapeSummary, setScrapeSummary] = useState<ScrapeSummaryData | null>(null)
  const [scrapeDetails, setScrapeDetails] = useState<ScrapeDetail[]>([])
  const [scrapeTopPosts, setScrapeTopPosts] = useState<ScrapeTopPost[]>([])
  const [redditOverview, setRedditOverview] = useState<RedditOverview | null>(null)
  const [redditBySubreddit, setRedditBySubreddit] = useState<RedditBySubredditData | null>(null)
  const [redditSignalBreakdown, setRedditSignalBreakdown] = useState<RedditSignalBreakdownData | null>(null)
  const [redditPerVendor, setRedditPerVendor] = useState<RedditPerVendorData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())
  const [days, setDays] = useState(30)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAdminDashboardData(days)
      setSummary(data.summary)
      setProviders(data.providers)
      setModels(data.models)
      setWorkflows(data.workflows)
      setDaily(data.daily)
      setRecent(data.recent)
      setReasoningActivity(data.reasoningActivity)
      setTasks(data.tasks)
      setErrorTimeline(data.errorTimeline)
      setSysResources(data.sysResources)
      setScrapeSummary(data.scrapeSummary)
      setScrapeDetails(data.scrapeDetails)
      setScrapeTopPosts(data.scrapeTopPosts)
      setRedditOverview(data.redditOverview)
      setRedditBySubreddit(data.redditBySubreddit)
      setRedditSignalBreakdown(data.redditSignalBreakdown)
      setRedditPerVendor(data.redditPerVendor)
      setLastRefresh(new Date())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }, [days])

  useEffect(() => { void fetchAll() }, [fetchAll])

  // Auto-refresh every 60s
  useEffect(() => {
    const id = setInterval(() => { void fetchAll() }, 60_000)
    return () => clearInterval(id)
  }, [fetchAll])

  // System resources fast refresh every 10s
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const resources = await fetchSystemResources()
        if (resources) setSysResources(resources)
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

  const sectionFallback = (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-5 text-sm text-slate-600">
      Loading panel...
    </div>
  )

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
          <div className="space-y-6">
            <ProviderChart data={providers} />
            <Suspense fallback={sectionFallback}>
              <ModelTable data={models} />
            </Suspense>
          </div>
        </div>

        {/* ── Task Health ─────────────────────────── */}
        {tasks.length > 0 && (
          <Suspense fallback={sectionFallback}>
            <TaskHealthPanel tasks={tasks} />
          </Suspense>
        )}

        {/* ── Scraping Pipeline ───────────────────── */}
        <Suspense fallback={sectionFallback}>
          <ScrapingPipeline
            summary={scrapeSummary}
            details={scrapeDetails}
            topPosts={scrapeTopPosts}
            redditOverview={redditOverview}
            redditBySubreddit={redditBySubreddit}
            redditSignalBreakdown={redditSignalBreakdown}
            redditPerVendor={redditPerVendor}
          />
        </Suspense>

        {/* ── Reasoning Activity ─────────────────── */}
        <div className="mb-8">
          <Suspense fallback={sectionFallback}>
            <ReasoningPanel reasoning={reasoningActivity} workflows={workflows} recent={recent} />
          </Suspense>
        </div>

        {/* ── Bottom: Workflows + Recent ─────────── */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Suspense fallback={sectionFallback}>
            <WorkflowTable data={workflows} />
          </Suspense>
          <Suspense fallback={sectionFallback}>
            <RecentCalls data={recent} />
          </Suspense>
        </div>

        {/* Footer */}
        <footer className="mt-10 border-t border-slate-800/50 pt-4 text-center text-[11px] text-slate-600">
          Atlas Brain &middot; Cost Monitor &middot; Auto-refreshes every 60s
        </footer>
      </div>
    </div>
  )
}
