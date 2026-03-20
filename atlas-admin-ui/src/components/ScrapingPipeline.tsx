import { Fragment, useState } from 'react'
import { Radar, Search, TrendingUp, MessageSquare, Shield, ExternalLink } from 'lucide-react'
import { fetchScrapeRunPages } from '../api/client'
import type {
  RedditBySubredditData,
  RedditOverview,
  RedditPerVendorData,
  RedditSignalBreakdownData,
  ScrapeSummaryData,
  ScrapeDetail,
  ScrapeRunPagesData,
  ScrapeTopPost,
} from '../types'
import { fmtDuration, timeAgo } from '../utils'

export default function ScrapingPipeline({
  summary,
  details,
  topPosts,
  redditOverview,
  redditBySubreddit,
  redditSignalBreakdown,
  redditPerVendor,
}: {
  summary: ScrapeSummaryData | null
  details: ScrapeDetail[]
  topPosts: ScrapeTopPost[]
  redditOverview: RedditOverview | null
  redditBySubreddit: RedditBySubredditData | null
  redditSignalBreakdown: RedditSignalBreakdownData | null
  redditPerVendor: RedditPerVendorData | null
}) {
  const [tab, setTab] = useState<'throughput' | 'details' | 'top-posts' | 'reddit'>('throughput')
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null)
  const [runPages, setRunPages] = useState<Record<string, ScrapeRunPagesData>>({})
  const [loadingRunId, setLoadingRunId] = useState<string | null>(null)
  const [runError, setRunError] = useState<string | null>(null)

  const handleToggleRun = async (detail: ScrapeDetail) => {
    if (!detail.has_page_logs) return
    if (expandedRunId === detail.id) {
      setExpandedRunId(null)
      return
    }
    setExpandedRunId(detail.id)
    setRunError(null)
    if (runPages[detail.id]) return
    setLoadingRunId(detail.id)
    try {
      const data = await fetchScrapeRunPages(detail.id)
      setRunPages((prev) => ({ ...prev, [detail.id]: data }))
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Failed to load page logs')
    } finally {
      setLoadingRunId(null)
    }
  }

  return (
    <div className="animate-enter mb-8" style={{ animationDelay: '400ms' }}>
      <div className="mb-4 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Radar className="h-4 w-4 text-cyan-400" />
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            Scraping Pipeline
          </h2>
        </div>
        {summary && (
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block h-2 w-2 rounded-full bg-cyan-400" />
              <span className="font-mono">{summary.today.runs}</span>
              <span className="text-slate-600">runs today</span>
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
              <span className="font-mono">{summary.today.reviews_inserted.toLocaleString()}</span>
              <span className="text-slate-600">inserted</span>
            </span>
            {summary.today.errors > 0 && (
              <span className="flex items-center gap-1.5 text-red-400">
                <span className="inline-block h-2 w-2 rounded-full bg-red-400" />
                <span className="font-mono">{summary.today.errors}</span>
                <span>errors</span>
              </span>
            )}
          </div>
        )}
      </div>

      {redditOverview && (
        <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <div className="rounded-lg border border-slate-800/70 bg-slate-950/50 px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-widest text-slate-600">Reddit Auth</div>
            <div className="mt-1 font-mono text-sm text-slate-200">{redditOverview.auth_mode}</div>
          </div>
          <div className="rounded-lg border border-slate-800/70 bg-slate-950/50 px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-widest text-slate-600">429 Events</div>
            <div className="mt-1 font-mono text-sm text-amber-400">{redditOverview.rate_limits.total_429_events}</div>
          </div>
          <div className="rounded-lg border border-slate-800/70 bg-slate-950/50 px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-widest text-slate-600">Triage Pass</div>
            <div className="mt-1 font-mono text-sm text-cyan-400">
              {(redditOverview.signal_funnel.triage_pass_rate * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-lg border border-slate-800/70 bg-slate-950/50 px-3 py-2.5">
            <div className="text-[10px] uppercase tracking-widest text-slate-600">Actionable</div>
            <div className="mt-1 font-mono text-sm text-emerald-400">
              {(redditOverview.signal_conversion.actionable_rate * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Sub-tabs */}
      <div className="mb-3 flex gap-1">
        {(['throughput', 'details', 'top-posts', 'reddit'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-lg px-3 py-1.5 text-[11px] font-medium transition-colors ${
              tab === t
                ? 'bg-cyan-500/15 text-cyan-400 ring-1 ring-cyan-500/25'
                : 'text-slate-500 hover:bg-slate-800/40 hover:text-slate-300'
            }`}
          >
            {t === 'throughput'
              ? 'Throughput'
              : t === 'details'
                ? 'Recent Scrapes'
                : t === 'top-posts'
                  ? 'Top Posts'
                  : 'Reddit Monitor'}
          </button>
        ))}
      </div>

      {tab === 'throughput' && <ThroughputTab summary={summary} />}
      {tab === 'details' && (
        <DetailsTab
          details={details}
          expandedRunId={expandedRunId}
          loadingRunId={loadingRunId}
          runError={runError}
          runPages={runPages}
          onToggleRun={handleToggleRun}
        />
      )}
      {tab === 'top-posts' && <TopPostsTab posts={topPosts} />}
      {tab === 'reddit' && (
        <RedditMonitoringTab
          overview={redditOverview}
          bySubreddit={redditBySubreddit}
          signalBreakdown={redditSignalBreakdown}
          perVendor={redditPerVendor}
        />
      )}
    </div>
  )
}

function ThroughputTab({ summary }: { summary: ScrapeSummaryData | null }) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* Signal quality cards */}
      {summary && summary.quality.length > 0 && (
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4 lg:col-span-1">
          <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
            Signal Quality by Source
          </h3>
          <div className="space-y-3">
            {summary.quality.map(q => (
              <div key={q.source} className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-xs font-medium text-slate-300">{q.source}</span>
                  <span className="font-mono text-[10px] text-slate-500">{q.total_reviews.toLocaleString()} reviews</span>
                </div>
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[10px]">
                    <span className="text-slate-500">High Signal ({'>'}0.7)</span>
                    <span className={`font-mono ${q.high_signal_rate > 0.3 ? 'text-emerald-400' : q.high_signal_rate > 0.1 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(q.high_signal_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full rounded-full ${q.high_signal_rate > 0.3 ? 'bg-emerald-400' : q.high_signal_rate > 0.1 ? 'bg-amber-400' : 'bg-red-400'}`}
                      style={{ width: `${Math.min(q.high_signal_rate * 100, 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-slate-500">Enriched</span>
                    <span className="font-mono text-cyan-400">{(q.enrichment_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
                    <div className="h-full rounded-full bg-cyan-400" style={{ width: `${Math.min(q.enrichment_rate * 100, 100)}%` }} />
                  </div>
                  <div className="mt-1 flex gap-3 text-[10px] text-slate-500">
                    <span>Avg weight: <span className="font-mono text-slate-400">{q.avg_source_weight.toFixed(3)}</span></span>
                    <span>HV authors: <span className="font-mono text-violet-400">{q.high_value_authors}</span></span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Throughput table */}
      <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4 lg:col-span-2">
        <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
          Throughput by Vendor
        </h3>
        <div className="max-h-[500px] overflow-y-auto">
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
              <tr className="border-b border-slate-800 text-[10px] uppercase tracking-widest text-slate-600">
                <th className="pb-2 pr-3 font-semibold">Vendor</th>
                <th className="pb-2 pr-3 text-right font-semibold">Runs</th>
                <th className="pb-2 pr-3 text-right font-semibold">Found</th>
                <th className="pb-2 pr-3 text-right font-semibold">Inserted</th>
                <th className="pb-2 pr-3 text-right font-semibold">Rate</th>
                <th className="pb-2 pr-3 text-right font-semibold">Avg Time</th>
                <th className="pb-2 text-right font-semibold">Captcha</th>
              </tr>
            </thead>
            <tbody>
              {(summary?.throughput ?? []).map((t, i) => (
                <tr key={i} className="border-b border-slate-800/40 transition-colors hover:bg-slate-800/20">
                  <td className="py-2 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-400" />
                      <span className="font-mono text-[11px] text-slate-300">{t.vendor_name}</span>
                    </div>
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-400">{t.total_runs}</td>
                  <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-400">{t.reviews_found.toLocaleString()}</td>
                  <td className="py-2 pr-3 text-right font-mono text-[11px] text-emerald-400">{t.reviews_inserted.toLocaleString()}</td>
                  <td className="py-2 pr-3 text-right">
                    <span className={`font-mono text-[11px] ${t.insert_rate > 0.3 ? 'text-emerald-400' : t.insert_rate > 0.1 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(t.insert_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-500">{fmtDuration(t.avg_duration_ms)}</td>
                  <td className="py-2 text-right font-mono text-[11px]">
                    {t.captcha_attempts > 0 ? (
                      <span className="text-amber-400">{t.captcha_attempts}</span>
                    ) : (
                      <span className="text-slate-600">0</span>
                    )}
                  </td>
                </tr>
              ))}
              {(summary?.throughput ?? []).length === 0 && (
                <tr><td colSpan={7} className="py-8 text-center text-sm text-slate-600">No scrape data yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function RedditMonitoringTab({
  overview,
  bySubreddit,
  signalBreakdown,
  perVendor,
}: {
  overview: RedditOverview | null
  bySubreddit: RedditBySubredditData | null
  signalBreakdown: RedditSignalBreakdownData | null
  perVendor: RedditPerVendorData | null
}) {
  const topSubreddits = (bySubreddit?.subreddits ?? []).slice(0, 8)
  const topVendors = (perVendor?.vendors ?? []).slice(0, 8)
  const topFlairs = (signalBreakdown?.flair_analysis ?? []).slice(0, 6)

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.15fr_0.85fr]">
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <MetricCard label="Runs" value={overview ? String(overview.runs.total) : '--'} />
          <MetricCard label="Insert Rate" value={overview ? pct(overview.throughput.insert_rate) : '--'} />
          <MetricCard label="Enrichment" value={overview ? pct(overview.signal_funnel.enrichment_completion_rate) : '--'} />
          <MetricCard label="Actionable" value={overview ? pct(overview.signal_conversion.actionable_rate) : '--'} />
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
          <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
            Top Subreddits by Yield
          </h3>
          <div className="max-h-[340px] overflow-y-auto">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
                <tr className="border-b border-slate-800 text-[10px] uppercase tracking-widest text-slate-600">
                  <th className="pb-2 pr-3 font-semibold">Subreddit</th>
                  <th className="pb-2 pr-3 text-right font-semibold">Posts</th>
                  <th className="pb-2 pr-3 text-right font-semibold">Signal</th>
                  <th className="pb-2 pr-3 text-right font-semibold">Triage</th>
                  <th className="pb-2 text-right font-semibold">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {topSubreddits.map((entry) => (
                  <tr key={entry.subreddit} className="border-b border-slate-800/40">
                    <td className="py-2 pr-3 font-mono text-[11px] text-slate-300">r/{entry.subreddit}</td>
                    <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-500">{entry.total_posts}</td>
                    <td className="py-2 pr-3 text-right font-mono text-[11px] text-cyan-400">{pct(entry.signal_rate)}</td>
                    <td className="py-2 pr-3 text-right font-mono text-[11px] text-emerald-400">{pct(entry.triage_pass_rate)}</td>
                    <td className="py-2 text-right font-mono text-[11px] text-amber-400">{entry.avg_urgency_score.toFixed(1)}</td>
                  </tr>
                ))}
                {topSubreddits.length === 0 && (
                  <tr><td colSpan={5} className="py-8 text-center text-sm text-slate-600">No Reddit subreddit data</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
          <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
            Vendor Signal Yield
          </h3>
          <div className="space-y-2">
            {topVendors.map((entry) => (
              <div key={entry.vendor_name} className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-3">
                <div className="flex items-start justify-between gap-3">
                  <span className="font-mono text-[11px] text-slate-200">{entry.vendor_name}</span>
                  <span className="font-mono text-[10px] text-cyan-400">{pct(entry.triage_pass_rate)}</span>
                </div>
                <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-500">
                  <span>{entry.inserted} inserted</span>
                  <span>{entry.enriched} enriched</span>
                  <span>{entry.intent_to_leave_count} leave</span>
                  <span>{entry.high_urgency_count} high urgency</span>
                </div>
                <div className="mt-2 text-[10px] text-slate-500">
                  Subs: <span className="text-slate-400">{entry.top_subreddits.join(', ') || 'n/a'}</span>
                </div>
                <div className="mt-1 text-[10px] text-slate-500">
                  Pain: <span className="text-slate-400">{entry.top_pain_categories.join(', ') || 'n/a'}</span>
                </div>
              </div>
            ))}
            {topVendors.length === 0 && (
              <div className="py-8 text-center text-sm text-slate-600">No vendor-level Reddit data</div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
          <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
            Signal Diagnostics
          </h3>
          <div className="grid grid-cols-2 gap-3 text-[11px]">
            <MetricCard label="Edited Signal" value={signalBreakdown ? pct(signalBreakdown.edit_stats.edited_signal_rate) : '--'} />
            <MetricCard label="Crosspost Signal" value={signalBreakdown ? pct(signalBreakdown.crosspost_stats.crosspost_signal_rate) : '--'} />
            <MetricCard label="Comment Trigger" value={signalBreakdown ? pct(signalBreakdown.comment_harvest_stats.comment_trigger_rate) : '--'} />
            <MetricCard label="Avg Churn" value={signalBreakdown ? signalBreakdown.author_churn_stats.avg_churn_score.toFixed(1) : '--'} />
          </div>
          <div className="mt-4 space-y-2">
            {topFlairs.map((entry) => (
              <div key={entry.flair} className="flex items-center justify-between gap-3 rounded-lg border border-slate-800/60 bg-slate-950/40 px-3 py-2">
                <span className="truncate text-[11px] text-slate-300">{entry.flair}</span>
                <div className="flex items-center gap-3 font-mono text-[10px]">
                  <span className="text-slate-500">{entry.count}</span>
                  <span className="text-cyan-400">{pct(entry.signal_rate)}</span>
                  <span className="text-slate-400">{entry.avg_weight.toFixed(2)}</span>
                </div>
              </div>
            ))}
            {topFlairs.length === 0 && (
              <div className="py-6 text-center text-sm text-slate-600">No flair diagnostics</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800/70 bg-slate-950/50 px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-widest text-slate-600">{label}</div>
      <div className="mt-1 font-mono text-sm text-slate-200">{value}</div>
    </div>
  )
}

function DetailsTab({
  details,
  expandedRunId,
  loadingRunId,
  runError,
  runPages,
  onToggleRun,
}: {
  details: ScrapeDetail[]
  expandedRunId: string | null
  loadingRunId: string | null
  runError: string | null
  runPages: Record<string, ScrapeRunPagesData>
  onToggleRun: (detail: ScrapeDetail) => Promise<void>
}) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
      <div className="max-h-[500px] overflow-y-auto">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
            <tr className="border-b border-slate-800 text-[10px] uppercase tracking-widest text-slate-600">
              <th className="pb-2 pr-3 font-semibold">Vendor</th>
              <th className="pb-2 pr-3 font-semibold">Status</th>
              <th className="pb-2 pr-3 text-right font-semibold">Found</th>
              <th className="pb-2 pr-3 text-right font-semibold">Inserted</th>
              <th className="pb-2 pr-3 text-right font-semibold">Rate</th>
              <th className="pb-2 pr-3 text-right font-semibold">Duration</th>
              <th className="pb-2 pr-3 font-semibold">Parser</th>
              <th className="pb-2 pr-3 font-semibold">Proxy</th>
              <th className="pb-2 font-semibold">When</th>
            </tr>
          </thead>
          <tbody>
            {details.map(d => (
              <Fragment key={d.id}>
              <tr
                className={`border-b border-slate-800/40 transition-colors hover:bg-slate-800/20 ${d.has_page_logs ? 'cursor-pointer' : ''}`}
                title={d.errors.length ? d.errors.join('\n') : undefined}
                onClick={() => { void onToggleRun(d) }}
              >
                <td className="py-2 pr-3 font-mono text-[11px] text-slate-300">{d.vendor_name}</td>
                <td className="py-2 pr-3">
                  <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                    d.status === 'success' || d.status === 'completed' ? 'bg-emerald-500/10 text-emerald-400'
                    : d.status === 'failed' ? 'bg-red-500/10 text-red-400'
                    : d.status === 'blocked' ? 'bg-amber-500/10 text-amber-400'
                    : 'bg-slate-500/10 text-slate-400'
                  }`}>
                    {d.status}
                  </span>
                </td>
                <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-400">{d.reviews_found}</td>
                <td className="py-2 pr-3 text-right font-mono text-[11px] text-emerald-400">{d.reviews_inserted}</td>
                <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-400">{(d.insert_rate * 100).toFixed(1)}%</td>
                <td className="py-2 pr-3 text-right font-mono text-[11px] text-slate-500">{fmtDuration(d.duration_ms)}</td>
                <td className="py-2 pr-3 font-mono text-[10px] text-slate-600">{d.parser_version}</td>
                <td className="py-2 pr-3 font-mono text-[10px] text-slate-600">{d.proxy_type || 'none'}</td>
                <td className="py-2 text-[10px] text-slate-500">{d.started_at ? timeAgo(d.started_at) : '--'}</td>
              </tr>
              {expandedRunId === d.id && (
                <tr className="border-b border-slate-800/40 bg-slate-950/60">
                  <td colSpan={9} className="px-4 py-3">
                    {loadingRunId === d.id && <div className="text-sm text-slate-500">Loading page logs...</div>}
                    {runError && <div className="text-sm text-red-400">{runError}</div>}
                    {runPages[d.id] && <RunPagesPanel data={runPages[d.id]} />}
                  </td>
                </tr>
              )}
              </Fragment>
            ))}
            {details.length === 0 && (
              <tr><td colSpan={9} className="py-8 text-center text-sm text-slate-600">No recent scrapes</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function RunPagesPanel({ data }: { data: ScrapeRunPagesData }) {
  const previewPages = data.pages.slice(0, 6)

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
        <span>{data.run.pages_scraped} pages</span>
        <span>{data.run.reviews_found} found</span>
        <span>{data.run.reviews_inserted} inserted</span>
        <span>{data.page_count} logs</span>
        {data.run.stop_reason && <span className="text-amber-400">{data.run.stop_reason}</span>}
      </div>
      <div className="space-y-2">
        {previewPages.map((page) => (
          <div key={`${data.run.id}-${page.page}`} className="rounded-lg border border-slate-800/60 bg-slate-900/40 px-3 py-2 text-[11px]">
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono text-slate-300">Page {page.page}</span>
              <span className="font-mono text-slate-500">{page.status_code ?? '--'}</span>
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-slate-500">
              <span>{page.reviews_parsed ?? 0} parsed</span>
              <span>{page.review_nodes_found ?? 0} nodes</span>
              <span>{fmtDuration(page.duration_ms ?? 0)}</span>
              {page.stop_reason && <span className="text-amber-400">{page.stop_reason}</span>}
            </div>
            {page.errors.length > 0 && (
              <div className="mt-1 truncate text-red-400/80">{page.errors.join(' | ')}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function TopPostsTab({ posts }: { posts: ScrapeTopPost[] }) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4">
      <div className="max-h-[560px] space-y-2 overflow-y-auto pr-1">
        {posts.map(p => (
          <div key={p.id} className="group rounded-lg border border-slate-800/60 bg-slate-950/40 p-3 transition-colors hover:border-slate-700/60 hover:bg-slate-800/20">
            <div className="mb-1.5 flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] font-medium text-cyan-400">{p.vendor_name}</span>
                  <span className="text-[10px] text-slate-600">&middot;</span>
                  <span className="font-mono text-[10px] text-violet-400">r/{p.subreddit}</span>
                  {p.post_flair && (
                    <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[9px] text-slate-500">{p.post_flair}</span>
                  )}
                </div>
                <p className="mt-1 truncate text-xs text-slate-300">{p.summary}</p>
              </div>
              <a
                href={p.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="shrink-0 rounded p-1 text-slate-600 opacity-0 transition-all hover:bg-slate-800 hover:text-cyan-400 group-hover:opacity-100"
              >
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px]">
              <span className="flex items-center gap-1 text-slate-500">
                <Search className="h-2.5 w-2.5" />
                weight: <span className={`font-mono ${p.source_weight >= 0.7 ? 'text-emerald-400' : 'text-slate-400'}`}>{p.source_weight.toFixed(2)}</span>
              </span>
              <span className="flex items-center gap-1 text-slate-500">
                <TrendingUp className="h-2.5 w-2.5" />
                <span className={`font-mono ${p.trending_score === 'high' ? 'text-red-400' : p.trending_score === 'medium' ? 'text-amber-400' : 'text-slate-500'}`}>
                  {p.trending_score}
                </span>
              </span>
              <span className="flex items-center gap-1 text-slate-500">
                <Shield className="h-2.5 w-2.5" />
                churn: <span className={`font-mono ${p.author_churn_score >= 7 ? 'text-red-400' : p.author_churn_score >= 3 ? 'text-amber-400' : 'text-slate-400'}`}>
                  {p.author_churn_score.toFixed(1)}
                </span>
              </span>
              <span className="text-slate-600">
                <span className="font-mono text-slate-500">{p.reddit_score}</span> pts
              </span>
              <span className="flex items-center gap-1 text-slate-600">
                <MessageSquare className="h-2.5 w-2.5" />
                <span className="font-mono text-slate-500">{p.num_comments}</span>
              </span>
              {p.is_edited && <span className="text-amber-500/60">edited</span>}
              {p.is_crosspost && <span className="text-violet-400/60">xpost</span>}
              {p.comment_count > 0 && (
                <span className="text-cyan-400/60">{p.comment_count} harvested</span>
              )}
              <span className={`rounded-full px-1.5 py-0 text-[9px] font-medium ${
                p.enrichment_status === 'completed' || p.enrichment_status === 'enriched'
                  ? 'bg-emerald-500/10 text-emerald-400'
                  : p.enrichment_status === 'failed' ? 'bg-red-500/10 text-red-400'
                  : 'bg-slate-500/10 text-slate-500'
              }`}>
                {p.enrichment_status}
              </span>
              <span className="text-slate-600">{p.reviewed_at ? timeAgo(p.reviewed_at) : '--'}</span>
            </div>
          </div>
        ))}
        {posts.length === 0 && (
          <p className="py-8 text-center text-sm text-slate-600">No high-signal posts found</p>
        )}
      </div>
    </div>
  )
}
