import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw, Swords, Shield, Zap, TrendingDown, Target, MessageSquareQuote } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import ArchetypeBadge from '../components/ArchetypeBadge'
import useApiData from '../hooks/useApiData'
import { fetchReport } from '../api/client'
import { REPORT_TYPE_COLORS } from './Reports'
import type { ReportDetail as ReportDetailType } from '../types'

function DetailSkeleton() {
  return (
    <div className="space-y-6 max-w-4xl animate-pulse">
      <div className="h-4 w-28 bg-slate-700/50 rounded" />
      <div>
        <div className="h-5 w-32 bg-slate-700/50 rounded mb-2" />
        <div className="h-7 w-56 bg-slate-700/50 rounded mb-2" />
        <div className="h-4 w-40 bg-slate-700/50 rounded" />
      </div>
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-32" />
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-48" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Structured renderers for intelligence_data fields
// ---------------------------------------------------------------------------

/** [{category: str, count: n}, ...] or [{name/feature/role/competitor: str, count: n}] */
function isRankedList(val: unknown): val is Record<string, unknown>[] {
  if (!Array.isArray(val) || val.length === 0) return false
  const first = val[0]
  return typeof first === 'object' && first !== null && 'count' in first
}

function RankedList({ items }: { items: Record<string, unknown>[] }) {
  const maxCount = Math.max(...items.map((i) => Number(i.count) || 0), 1)
  return (
    <div className="space-y-1.5">
      {items.map((item, idx) => {
        const label = String(
          item.category ?? item.name ?? item.feature ?? item.role ?? item.competitor ?? `#${idx + 1}`,
        )
        const count = Number(item.count) || 0
        const pct = Math.round((count / maxCount) * 100)
        return (
          <div key={idx}>
            <div className="flex items-center justify-between text-xs mb-0.5">
              <span className="text-slate-300">{label}</span>
              <span className="text-slate-400">{count}</span>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-500/60 rounded-full"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** string[] -- render as blockquotes for quotes, pills otherwise */
function StringList({ items, asQuotes }: { items: string[]; asQuotes?: boolean }) {
  if (asQuotes) {
    return (
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {items.map((q, i) => (
          <blockquote
            key={i}
            className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3"
          >
            {q}
          </blockquote>
        ))}
      </div>
    )
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((s, i) => (
        <span key={i} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
          {s}
        </span>
      ))}
    </div>
  )
}

/** {key: number} flat object -- render as key-value grid */
function StatObject({ obj }: { obj: Record<string, unknown> }) {
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
      {Object.entries(obj).map(([k, v]) => (
        <div key={k} className="flex justify-between">
          <dt className="text-slate-400">{k.replace(/_/g, ' ')}</dt>
          <dd className="text-white">{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Render an array of objects as a responsive table */
function DataTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (rows.length === 0) return null
  // Collect all keys across all rows, but skip very long arrays/objects
  const allKeys = Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  // Filter out keys whose values are complex (arrays/deep objects) -- show simple values only
  const columns = allKeys.filter(k =>
    rows.some(r => {
      const v = r[k]
      return v !== null && v !== undefined && typeof v !== 'object'
    })
  )
  if (columns.length === 0) return null

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700/50">
            {columns.map(col => (
              <th key={col} className="text-left text-xs font-medium text-slate-400 uppercase tracking-wider px-3 py-2 whitespace-nowrap">
                {col.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
              {columns.map(col => {
                const val = row[col]
                // Archetype badge rendering
                if (col === 'archetype' && typeof val === 'string' && val) {
                  return (
                    <td key={col} className="px-3 py-2 whitespace-nowrap">
                      <ArchetypeBadge archetype={val} confidence={row.archetype_confidence as number} showConfidence />
                    </td>
                  )
                }
                // Skip archetype_confidence as standalone column (shown inside badge)
                if (col === 'archetype_confidence' && row.archetype) {
                  return <td key={col} className="px-3 py-2 text-slate-500 whitespace-nowrap">--</td>
                }
                // Risk level color coding
                if (col === 'risk_level' && typeof val === 'string') {
                  const riskColor: Record<string, string> = {
                    critical: 'text-red-400', high: 'text-orange-400',
                    medium: 'text-yellow-400', low: 'text-green-400',
                  }
                  return (
                    <td key={col} className="px-3 py-2 whitespace-nowrap">
                      <span className={`text-sm font-medium ${riskColor[val] ?? 'text-slate-300'}`}>{val}</span>
                    </td>
                  )
                }
                const display = val === null || val === undefined ? '--'
                  : typeof val === 'number' ? (Number.isInteger(val) ? String(val) : val.toFixed(1))
                  : String(val)
                return (
                  <td key={col} className="px-3 py-2 text-slate-300 whitespace-nowrap">
                    {display}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Challenger Brief — dedicated renderer
// ---------------------------------------------------------------------------

type AnyObj = Record<string, any>

function CBSection({ title, icon, children }: { title: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
      <h3 className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-3">
        {icon}
        {title}
      </h3>
      {children}
    </div>
  )
}

function CBMetric({ label, value, color }: { label: string; value: any; color?: string }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div className="flex justify-between text-sm">
      <span className="text-slate-400">{label}</span>
      <span className={color ?? 'text-white'}>{String(value)}</span>
    </div>
  )
}

function pctFmt(v: any): string {
  if (typeof v !== 'number') return '--'
  return `${(v * 100).toFixed(1)}%`
}

function riskColor(level: string | undefined): string {
  const m: Record<string, string> = { critical: 'text-red-400', high: 'text-orange-400', medium: 'text-yellow-400', low: 'text-green-400' }
  return m[level ?? ''] ?? 'text-slate-300'
}

function ChallengerBriefDetail({ data }: { data: AnyObj }) {
  const disp: AnyObj = data.displacement_summary ?? {}
  const inc: AnyObj = data.incumbent_profile ?? {}
  const adv: AnyObj = data.challenger_advantage ?? {}
  const h2h: AnyObj = data.head_to_head ?? {}
  const targets: AnyObj[] = data.target_accounts ?? []
  const playbook: AnyObj = data.sales_playbook ?? {}
  const integ: AnyObj = data.integration_comparison ?? {}
  const sources: AnyObj = data.data_sources ?? {}

  return (
    <div className="space-y-6">
      {/* Data source pills */}
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(sources).map(([k, v]) => (
          <span
            key={k}
            className={clsx(
              'px-2 py-0.5 rounded text-xs font-medium',
              v ? 'bg-cyan-900/40 text-cyan-300' : 'bg-slate-800 text-slate-600',
            )}
          >
            {k.replace(/_/g, ' ')}
          </span>
        ))}
      </div>

      {/* Displacement Signal */}
      <CBSection title="Displacement Signal" icon={<TrendingDown className="h-4 w-4 text-red-400" />}>
        <div className="space-y-1">
          <CBMetric label="Total Mentions" value={disp.total_mentions} color={Number(disp.total_mentions) >= 50 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />
          <CBMetric label="Signal Strength" value={disp.signal_strength} />
          <CBMetric label="Confidence" value={typeof disp.confidence_score === 'number' ? pctFmt(disp.confidence_score) : disp.confidence_score} />
          <CBMetric label="Primary Driver" value={disp.primary_driver} />
        </div>
        {disp.source_distribution && typeof disp.source_distribution === 'object' && Object.keys(disp.source_distribution).length > 0 && (
          <p className="text-xs text-slate-500 mt-2">
            Sources: {Object.entries(disp.source_distribution as Record<string, number>)
              .sort(([, a], [, b]) => b - a)
              .map(([src, cnt]) => `${src}: ${cnt}`)
              .join(', ')}
          </p>
        )}
        {disp.key_quote && (
          <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 mt-2">
            "{disp.key_quote}"
          </blockquote>
        )}
      </CBSection>

      {/* Incumbent Profile */}
      {Object.keys(inc).length > 0 && (
        <CBSection title={`Incumbent: ${data.incumbent}`} icon={<Shield className="h-4 w-4 text-red-400" />}>
          <div className="space-y-1">
            {inc.archetype && (
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Archetype</span>
                <ArchetypeBadge archetype={inc.archetype} confidence={inc.archetype_confidence} showConfidence />
              </div>
            )}
            <CBMetric label="Risk Level" value={inc.risk_level} color={riskColor(inc.risk_level)} />
            {inc.churn_pressure_score != null && (
              <CBMetric label="Churn Pressure" value={`${Number(inc.churn_pressure_score).toFixed(0)}/100`} color={Number(inc.churn_pressure_score) >= 60 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />
            )}
            <CBMetric label="Price Complaint Rate" value={pctFmt(inc.price_complaint_rate)} />
            <CBMetric label="DM Churn Rate" value={pctFmt(inc.dm_churn_rate)} />
            <CBMetric label="Sentiment Trend" value={inc.sentiment_direction} />
          </div>

          {Array.isArray(inc.key_signals) && inc.key_signals.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Key Signals</p>
              <ul className="space-y-0.5">
                {inc.key_signals.slice(0, 5).map((s: string, i: number) => (
                  <li key={i} className="text-xs text-slate-300 flex gap-2">
                    <span className="text-amber-400">-</span> {s}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {Array.isArray(inc.top_weaknesses) && inc.top_weaknesses.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Top Weaknesses</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left text-xs text-slate-400 px-2 py-1">Weakness</th>
                      <th className="text-right text-xs text-slate-400 px-2 py-1">Evidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inc.top_weaknesses.slice(0, 8).map((w: AnyObj, i: number) => (
                      <tr key={i} className="border-b border-slate-800/50">
                        <td className="px-2 py-1 text-slate-300">{w.area ?? w.weakness ?? w.name ?? ''}</td>
                        <td className="px-2 py-1 text-slate-400 text-right">{w.count ?? w.evidence_count ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {Array.isArray(inc.top_pain_quotes) && inc.top_pain_quotes.length > 0 && (
            <div className="space-y-2 mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Customer Pain</p>
              {inc.top_pain_quotes.slice(0, 5).map((q: any, i: number) => (
                <blockquote key={i} className="text-sm text-slate-300 italic border-l-2 border-red-500/50 pl-3">
                  "{typeof q === 'string' ? q : q?.quote ?? ''}"
                  {q?.source_site && <span className="text-xs text-slate-500 not-italic ml-2">({q.source_site})</span>}
                </blockquote>
              ))}
            </div>
          )}
        </CBSection>
      )}

      {/* Challenger Advantage */}
      {(Array.isArray(adv.strengths) && adv.strengths.length > 0 || adv.profile_summary) && (
        <CBSection title={`Challenger: ${data.challenger}`} icon={<Zap className="h-4 w-4 text-green-400" />}>
          {adv.profile_summary && <p className="text-sm text-slate-300 mb-3">{adv.profile_summary}</p>}

          {Array.isArray(adv.strengths) && adv.strengths.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-400 px-2 py-1">Strength</th>
                    <th className="text-right text-xs text-slate-400 px-2 py-1">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {adv.strengths.slice(0, 8).map((s: AnyObj, i: number) => (
                    <tr key={i} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300">{s.area ?? s.name ?? ''}</td>
                      <td className="px-2 py-1 text-slate-400 text-right">{s.evidence_count ?? s.mentions ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {Array.isArray(adv.weakness_coverage) && adv.weakness_coverage.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Weakness Coverage</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left text-xs text-slate-400 px-2 py-1">Incumbent Weakness</th>
                      <th className="text-left text-xs text-slate-400 px-2 py-1">Match</th>
                    </tr>
                  </thead>
                  <tbody>
                    {adv.weakness_coverage.slice(0, 8).map((c: AnyObj, i: number) => (
                      <tr key={i} className="border-b border-slate-800/50">
                        <td className="px-2 py-1 text-slate-300">{c.incumbent_weakness ?? ''}</td>
                        <td className="px-2 py-1">
                          <span className={clsx('text-xs px-1.5 py-0.5 rounded',
                            c.match_quality === 'strong' ? 'bg-green-500/15 text-green-300' : 'bg-amber-500/15 text-amber-300'
                          )}>{c.match_quality}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {Array.isArray(adv.commonly_switched_from) && adv.commonly_switched_from.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Commonly Switched From</p>
              <div className="flex flex-wrap gap-1.5">
                {adv.commonly_switched_from.slice(0, 10).map((sf: any, i: number) => (
                  <span key={i} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                    {typeof sf === 'string' ? sf : sf.vendor ?? ''}
                  </span>
                ))}
              </div>
            </div>
          )}
        </CBSection>
      )}

      {/* Head to Head */}
      {(h2h.conclusion || h2h.winner) && (
        <CBSection title="Head to Head" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
          <div className="space-y-1">
            {h2h.winner && <CBMetric label="Winner" value={h2h.winner} color="text-cyan-400 font-bold" />}
            {h2h.confidence != null && <CBMetric label="Confidence" value={pctFmt(h2h.confidence)} />}
            {h2h.durability && <CBMetric label="Durability" value={h2h.durability} />}
            {h2h.synthesized && <span className="text-[10px] text-slate-600">(synthesized from displacement data)</span>}
          </div>
          {h2h.conclusion && <p className="text-sm text-slate-300 mt-2">{h2h.conclusion}</p>}
          {Array.isArray(h2h.key_insights) && h2h.key_insights.length > 0 && (
            <ul className="space-y-1 mt-2">
              {h2h.key_insights.slice(0, 5).map((ins: any, i: number) => {
                const text = typeof ins === 'string' ? ins : ins?.insight ?? ''
                const evidence = typeof ins === 'object' ? ins?.evidence : ''
                return (
                  <li key={i} className="text-xs text-slate-400 flex gap-2">
                    <span className="text-cyan-400">-</span>
                    <span>{text}{evidence && <span className="text-slate-600 ml-1">({evidence})</span>}</span>
                  </li>
                )
              })}
            </ul>
          )}
        </CBSection>
      )}

      {/* Target Accounts */}
      {targets.length > 0 && (
        <CBSection
          title={`Target Accounts (${data.total_target_accounts ?? targets.length} total, ${data.accounts_considering_challenger ?? 0} considering ${data.challenger})`}
          icon={<Target className="h-4 w-4 text-amber-400" />}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/50">
                  {['Company', 'Score', 'Stage', 'Urg', 'Industry', 'Chall?'].map(h => (
                    <th key={h} className="text-left text-xs text-slate-400 px-2 py-1 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {targets.slice(0, 15).map((t: AnyObj, i: number) => (
                  <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="px-2 py-1 text-slate-300">{t.company ?? ''}</td>
                    <td className="px-2 py-1 text-white font-medium">{t.opportunity_score ?? ''}</td>
                    <td className="px-2 py-1 text-slate-300">{(t.buying_stage ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-300">{typeof t.urgency === 'number' ? t.urgency.toFixed(0) : ''}</td>
                    <td className="px-2 py-1 text-slate-400">{(t.industry ?? '').slice(0, 20)}</td>
                    <td className="px-2 py-1">{t.considers_challenger ? <span className="text-green-400">Y</span> : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CBSection>
      )}

      {/* Sales Playbook */}
      {Object.keys(playbook).length > 0 && (
        <CBSection title="Sales Playbook" icon={<MessageSquareQuote className="h-4 w-4 text-cyan-400" />}>
          {Array.isArray(playbook.discovery_questions) && playbook.discovery_questions.length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Discovery Questions</p>
              <ol className="space-y-1.5 list-none">
                {playbook.discovery_questions.slice(0, 5).map((q: string, i: number) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span className="shrink-0 w-5 h-5 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-xs">{i + 1}</span>
                    <span className="text-slate-300">{q}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {Array.isArray(playbook.landmine_questions) && playbook.landmine_questions.length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Landmine Questions</p>
              <ol className="space-y-1.5 list-none">
                {playbook.landmine_questions.slice(0, 3).map((q: string, i: number) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span className="shrink-0 w-5 h-5 rounded-full bg-red-500/15 text-red-400 flex items-center justify-center text-xs">{i + 1}</span>
                    <span className="text-slate-300">{q}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {Array.isArray(playbook.objection_handlers) && playbook.objection_handlers.length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Objection Handlers</p>
              <div className="space-y-2">
                {playbook.objection_handlers.map((h: AnyObj, i: number) => (
                  <div key={i} className="bg-slate-800/50 rounded-lg p-3">
                    {h.objection && <p className="text-sm text-red-300 mb-1"><span className="text-xs text-slate-500 uppercase mr-1">Objection:</span>{h.objection}</p>}
                    {h.pivot && <p className="text-sm text-green-300"><span className="text-xs text-slate-500 uppercase mr-1">Pivot:</span>{h.pivot}</p>}
                    {h.proof_point && <p className="text-xs text-slate-400 mt-1">{h.proof_point}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {playbook.talk_track && typeof playbook.talk_track === 'object' && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Talk Track</p>
              <div className="space-y-3">
                {[
                  { key: 'opening', label: 'Opening', color: 'border-cyan-500/50' },
                  { key: 'mid_call_pivot', label: 'Mid-Call Pivot', color: 'border-amber-500/50' },
                  { key: 'closing', label: 'Closing', color: 'border-green-500/50' },
                ].map(({ key, label, color }) => {
                  const val = (playbook.talk_track as AnyObj)[key]
                  if (!val) return null
                  return (
                    <div key={key} className={`border-l-2 ${color} pl-3`}>
                      <p className="text-xs text-slate-500 uppercase mb-1">{label}</p>
                      <p className="text-sm text-slate-300">{String(val)}</p>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {Array.isArray(playbook.recommended_plays) && playbook.recommended_plays.length > 0 && (
            <div>
              <p className="text-xs font-medium text-slate-400 mb-2">Recommended Plays</p>
              <div className="space-y-2">
                {playbook.recommended_plays.map((p: AnyObj, i: number) => (
                  <div key={i} className="bg-slate-800/50 rounded-lg p-3">
                    <p className="text-sm text-white font-medium">{p.play ?? p.name ?? p.description ?? ''}</p>
                    {p.target_segment && <span className="text-xs text-amber-300">{p.target_segment}</span>}
                    {p.key_message && <p className="text-xs text-slate-400 mt-1">{p.key_message}</p>}
                    {p.timing && <p className="text-xs text-slate-500 mt-1">Timing: {p.timing}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </CBSection>
      )}

      {/* Integration Comparison */}
      {integ && (Array.isArray(integ.shared) && integ.shared.length > 0 || Array.isArray(integ.challenger_exclusive) && integ.challenger_exclusive.length > 0) && (
        <CBSection title="Integration Comparison">
          <div className="space-y-2">
            {Array.isArray(integ.shared) && integ.shared.length > 0 && (
              <div>
                <span className="text-xs text-slate-400">Shared: </span>
                <span className="text-sm text-slate-300">{integ.shared.slice(0, 10).join(', ')}</span>
              </div>
            )}
            {Array.isArray(integ.challenger_exclusive) && integ.challenger_exclusive.length > 0 && (
              <div>
                <span className="text-xs text-slate-400">{data.challenger} Exclusive: </span>
                <span className="text-sm text-green-300">{integ.challenger_exclusive.slice(0, 10).join(', ')}</span>
              </div>
            )}
            {Array.isArray(integ.incumbent_exclusive) && integ.incumbent_exclusive.length > 0 && (
              <div>
                <span className="text-xs text-slate-400">{data.incumbent} Exclusive: </span>
                <span className="text-sm text-red-300">{integ.incumbent_exclusive.slice(0, 10).join(', ')}</span>
              </div>
            )}
          </div>
        </CBSection>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Accounts In Motion — dedicated renderer
// ---------------------------------------------------------------------------

function AccountsInMotionDetail({ data }: { data: AnyObj }) {
  const pricing: AnyObj = data.pricing_pressure ?? {}
  const gaps: AnyObj[] = Array.isArray(data.feature_gaps) ? data.feature_gaps : []
  const xvc: AnyObj = data.cross_vendor_context ?? {}
  const accounts: AnyObj[] = Array.isArray(data.accounts) ? data.accounts : []

  return (
    <div className="space-y-6">
      {/* Header metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Total Accounts</p>
          <p className="text-2xl font-bold text-cyan-400">{data.total_accounts_in_motion ?? accounts.length}</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Archetype</p>
          {data.archetype ? (
            <ArchetypeBadge archetype={data.archetype} confidence={data.archetype_confidence} showConfidence />
          ) : (
            <p className="text-sm text-slate-500">--</p>
          )}
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Price Complaint Rate</p>
          <p className="text-xl font-bold text-white">{pricing.price_complaint_rate != null ? pctFmt(pricing.price_complaint_rate) : '--'}</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Top Destination</p>
          <p className="text-sm font-medium text-white">{xvc.top_destination ?? '--'}</p>
        </div>
      </div>

      {/* Pricing Pressure + Feature Gaps side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pricing */}
        {(pricing.price_complaint_rate != null || pricing.price_increase_rate != null) && (
          <CBSection title="Pricing Pressure" icon={<TrendingDown className="h-4 w-4 text-amber-400" />}>
            <div className="space-y-1">
              <CBMetric label="Price Complaint Rate" value={pctFmt(pricing.price_complaint_rate)} />
              <CBMetric label="Price Increase Rate" value={pctFmt(pricing.price_increase_rate)} />
              {pricing.avg_seat_count != null && <CBMetric label="Avg Seat Count" value={Math.round(pricing.avg_seat_count)} />}
            </div>
          </CBSection>
        )}

        {/* Feature Gaps */}
        {gaps.length > 0 && (
          <CBSection title="Top Feature Gaps">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-400 px-2 py-1">Feature</th>
                    <th className="text-right text-xs text-slate-400 px-2 py-1">Mentions</th>
                  </tr>
                </thead>
                <tbody>
                  {gaps.slice(0, 10).map((g, i) => (
                    <tr key={i} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300">{g.feature ?? ''}</td>
                      <td className="px-2 py-1 text-slate-400 text-right">{g.mentions ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CBSection>
        )}
      </div>

      {/* Competitive Context */}
      {(xvc.top_destination || xvc.battle_conclusion || xvc.market_regime) && (
        <CBSection title="Competitive Context" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
          <div className="space-y-1">
            <CBMetric label="Top Destination" value={xvc.top_destination} color="text-cyan-400" />
            <CBMetric label="Market Regime" value={(xvc.market_regime ?? '').replace(/_/g, ' ')} />
          </div>
          {xvc.battle_conclusion && <p className="text-sm text-slate-300 mt-2">{xvc.battle_conclusion}</p>}
        </CBSection>
      )}

      {/* Accounts table */}
      {accounts.length > 0 && (
        <CBSection title={`Prospecting List (${accounts.length} accounts)`} icon={<Target className="h-4 w-4 text-amber-400" />}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/50">
                  {['Company', 'Score', 'Stage', 'Urg', 'Pain', 'Industry', 'Domain', 'Alts'].map(h => (
                    <th key={h} className="text-left text-xs text-slate-400 px-2 py-1 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {accounts.slice(0, 25).map((a, i) => (
                  <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="px-2 py-1 text-slate-300">{(a.company ?? '').slice(0, 24)}</td>
                    <td className="px-2 py-1">
                      <span className={clsx('font-medium', Number(a.opportunity_score) >= 50 ? 'text-green-400' : Number(a.opportunity_score) >= 30 ? 'text-amber-400' : 'text-slate-300')}>
                        {a.opportunity_score ?? ''}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-slate-300">{(a.buying_stage ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-300">{typeof a.urgency === 'number' ? a.urgency.toFixed(0) : ''}</td>
                    <td className="px-2 py-1 text-slate-400">{(a.pain_category ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-400">{(a.industry ?? '').slice(0, 18)}</td>
                    <td className="px-2 py-1 text-slate-400">{(a.domain ?? '').slice(0, 22)}</td>
                    <td className="px-2 py-1 text-xs text-slate-500">{Array.isArray(a.alternatives_considering) ? a.alternatives_considering.slice(0, 2).join(', ') : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Key quotes from top accounts */}
          {accounts.filter(a => a.top_quote).length > 0 && (
            <div className="mt-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Key Quotes</p>
              <div className="space-y-2">
                {accounts.filter(a => a.top_quote).slice(0, 5).map((a, i) => (
                  <div key={i}>
                    <span className="text-[10px] text-slate-500">{a.company}{a.urgency ? ` (urgency: ${a.urgency})` : ''}</span>
                    <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3">
                      "{a.top_quote}"
                    </blockquote>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CBSection>
      )}
    </div>
  )
}

/** Known scalar metadata keys that should render inline, not in a card */
const SCALAR_KEYS = new Set([
  'vendor_name', 'challenger_name', 'primary_vendor', 'comparison_vendor', 'report_date', 'window_days',
  'signal_count', 'high_urgency_count', 'medium_urgency_count',
  'scope', 'llm_model', 'model_analysis', 'parse_fallback',
])

const QUOTE_KEYS = new Set(['anonymized_quotes', 'quotable_evidence'])

/** Human-readable labels for battle card / reasoning fields */
const FIELD_LABELS: Record<string, string> = {
  avg_urgency: 'Avg Urgency',
  dm_churn_rate: 'DM Churn Rate',
  total_reviews: 'Total Reviews',
  churn_signal_density: 'Churn Signal Density',
  price_complaint_rate: 'Price Complaint Rate',
  sentiment_direction: 'Sentiment Direction',
  avg_seat_count: 'Avg Seat Count',
  max_seat_count: 'Max Seat Count',
  median_seat_count: 'Median Seat Count',
  price_increase_rate: 'Price Increase Rate',
  price_increase_count: 'Price Increase Count',
  market_structure: 'Market Structure',
  dominant_archetype: 'Dominant Archetype',
  displacement_intensity: 'Displacement Intensity',
  top_feature_gaps: 'Feature Gaps',
  budget_context: 'Budget Signals',
  ecosystem_context: 'Ecosystem',
  objection_data: 'Objection Data',
}

function humanLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return '--'
  if (typeof val === 'number') {
    if (val > 0 && val < 1) return `${(val * 100).toFixed(1)}%`
    if (Number.isInteger(val)) return val.toLocaleString()
    return val.toFixed(1)
  }
  return String(val)
}

/** Render a mixed object with scalars, nested objects, and arrays as a readable card */
function MixedObjectCard({ obj, label }: { obj: Record<string, unknown>; label?: string }) {
  const scalars: [string, unknown][] = []
  const nested: [string, unknown][] = []

  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue
    if (typeof v === 'object') {
      nested.push([k, v])
    } else {
      scalars.push([k, v])
    }
  }

  return (
    <div className="space-y-3">
      {label && <h5 className="text-xs font-medium text-cyan-400 uppercase tracking-wider">{label}</h5>}
      {scalars.length > 0 && (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
          {scalars.map(([k, v]) => (
            <div key={k} className="flex justify-between col-span-1">
              <dt className="text-slate-400">{humanLabel(k)}</dt>
              <dd className="text-white font-medium">{formatValue(v)}</dd>
            </div>
          ))}
        </dl>
      )}
      {nested.map(([k, v]) => {
        if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'object') {
          return (
            <div key={k}>
              <h6 className="text-xs text-slate-400 mb-1.5">{humanLabel(k)}</h6>
              <div className="space-y-1">
                {(v as Record<string, unknown>[]).map((item, i) => {
                  const name = String(item.feature ?? item.name ?? item.area ?? `#${i + 1}`)
                  const count = Number(item.mentions ?? item.count ?? 0)
                  return (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">{name}</span>
                      {count > 0 && <span className="text-xs text-slate-500">{count} mentions</span>}
                    </div>
                  )
                })}
              </div>
            </div>
          )
        }
        if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
          return <MixedObjectCard key={k} obj={v as Record<string, unknown>} label={humanLabel(k)} />
        }
        return null
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Battle card field renderers
// ---------------------------------------------------------------------------

function CompetitorDifferentiators({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="space-y-3">
      {items.map((c, i) => (
        <div key={i} className="bg-slate-800/50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-white font-medium text-sm">{String(c.competitor ?? '')}</span>
            <span className="text-xs text-slate-400">{Number(c.mentions ?? 0)} mentions</span>
          </div>
          <div className="flex gap-2 text-xs">
            {typeof c.primary_driver === 'string' && (
              <span className="px-2 py-0.5 bg-amber-500/15 text-amber-300 rounded">
                {c.primary_driver}
              </span>
            )}
            {typeof c.solves_weakness === 'string' && (
              <span className="px-2 py-0.5 bg-green-500/15 text-green-300 rounded">
                solves: {c.solves_weakness}
              </span>
            )}
            {Number(c.switch_count ?? 0) > 0 && (
              <span className="text-slate-400">{String(c.switch_count)} switches</span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function PainQuotes({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="space-y-3">
      {items.map((q, i) => (
        <div key={i} className="border-l-2 border-cyan-500/50 pl-3">
          <blockquote className="text-sm text-slate-300 italic">
            "{String(q.quote ?? '')}"
          </blockquote>
          <div className="flex gap-3 mt-1 text-xs text-slate-500">
            {typeof q.company === 'string' && <span>{q.company}</span>}
            {typeof q.title === 'string' && <span>{q.title}</span>}
            {typeof q.industry === 'string' && <span>{q.industry}</span>}
            {typeof q.source_site === 'string' && <span>{q.source_site}</span>}
            {Number(q.urgency ?? 0) > 0 && (
              <span className="text-amber-400">urgency {String(q.urgency)}/10</span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function ObjectionHandlers({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="space-y-3">
      {items.map((h, i) => (
        <div key={i} className="bg-slate-800/50 rounded-lg p-3">
          {typeof h.objection === 'string' && (
            <p className="text-sm text-red-300 mb-1.5">
              <span className="text-xs text-slate-500 uppercase mr-1">Objection:</span>
              {h.objection}
            </p>
          )}
          {typeof h.pivot === 'string' && (
            <p className="text-sm text-green-300">
              <span className="text-xs text-slate-500 uppercase mr-1">Pivot:</span>
              {h.pivot}
            </p>
          )}
        </div>
      ))}
    </div>
  )
}

function TalkTrack({ obj }: { obj: Record<string, unknown> }) {
  const sections = [
    { key: 'opening', label: 'Opening', color: 'border-cyan-500/50' },
    { key: 'proof_points', label: 'Proof Points', color: 'border-amber-500/50' },
    { key: 'closing', label: 'Closing', color: 'border-green-500/50' },
  ]
  return (
    <div className="space-y-3">
      {sections.map(({ key, label, color }) => {
        const val = obj[key]
        if (!val) return null
        if (typeof val === 'string') {
          return (
            <div key={key} className={`border-l-2 ${color} pl-3`}>
              <p className="text-xs text-slate-500 uppercase mb-1">{label}</p>
              <p className="text-sm text-slate-300">{val}</p>
            </div>
          )
        }
        if (Array.isArray(val)) {
          return (
            <div key={key} className={`border-l-2 ${color} pl-3`}>
              <p className="text-xs text-slate-500 uppercase mb-1">{label}</p>
              <ul className="space-y-1">
                {val.map((item, j) => (
                  <li key={j} className="text-sm text-slate-300">{String(item)}</li>
                ))}
              </ul>
            </div>
          )
        }
        return null
      })}
    </div>
  )
}

function WeaknessAnalysis({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="space-y-3">
      {items.map((w, i) => (
        <div key={i} className="bg-slate-800/50 rounded-lg p-3">
          <p className="text-sm text-white font-medium mb-1">{String(w.weakness ?? w.area ?? '')}</p>
          {typeof w.evidence === 'string' && (
            <p className="text-sm text-slate-400">{w.evidence}</p>
          )}
          {typeof w.recommendation === 'string' && (
            <p className="text-sm text-cyan-300 mt-1">{w.recommendation}</p>
          )}
        </div>
      ))}
    </div>
  )
}

function RecommendedPlays({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="space-y-3">
      {items.map((p, i) => (
        <div key={i} className="bg-slate-800/50 rounded-lg p-3">
          <p className="text-sm text-slate-300">{String(p.play ?? p.description ?? '')}</p>
          {typeof p.timing === 'string' && (
            <p className="text-xs text-slate-500 mt-1">Timing: {p.timing}</p>
          )}
        </div>
      ))}
    </div>
  )
}

function NumberedList({ items }: { items: string[] }) {
  return (
    <ol className="space-y-2 list-none">
      {items.map((q, i) => (
        <li key={i} className="flex gap-3 text-sm">
          <span className="shrink-0 w-6 h-6 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-xs font-medium">
            {i + 1}
          </span>
          <span className="text-slate-300 pt-0.5">{q}</span>
        </li>
      ))}
    </ol>
  )
}

function IntelValue({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  // Battle card structured objects
  if (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    (fieldKey === 'objection_data' || fieldKey === 'ecosystem_context' || fieldKey === 'budget_context')
  ) {
    return <MixedObjectCard obj={value as Record<string, unknown>} />
  }

  // Talk track (object with opening/closing/proof_points)
  if (fieldKey === 'talk_track' && typeof value === 'object' && value !== null && !Array.isArray(value)) {
    return <TalkTrack obj={value as Record<string, unknown>} />
  }

  // Competitive landscape (object with text fields)
  if (fieldKey === 'competitive_landscape' && typeof value === 'object' && value !== null && !Array.isArray(value)) {
    return <MixedObjectCard obj={value as Record<string, unknown>} />
  }

  // Array-based battle card fields
  if (Array.isArray(value) && value.length > 0) {
    if (fieldKey === 'competitor_differentiators' && typeof value[0] === 'object') {
      return <CompetitorDifferentiators items={value as Record<string, unknown>[]} />
    }
    if (fieldKey === 'customer_pain_quotes' && typeof value[0] === 'object') {
      return <PainQuotes items={value as Record<string, unknown>[]} />
    }
    if (fieldKey === 'objection_handlers' && typeof value[0] === 'object') {
      return <ObjectionHandlers items={value as Record<string, unknown>[]} />
    }
    if ((fieldKey === 'weakness_analysis' || fieldKey === 'vendor_weaknesses') && typeof value[0] === 'object') {
      return <WeaknessAnalysis items={value as Record<string, unknown>[]} />
    }
    if (fieldKey === 'recommended_plays' && typeof value[0] === 'object') {
      return <RecommendedPlays items={value as Record<string, unknown>[]} />
    }
    if ((fieldKey === 'discovery_questions' || fieldKey === 'landmine_questions') && typeof value[0] === 'string') {
      return <NumberedList items={value as string[]} />
    }
  }

  // String
  if (typeof value === 'string') {
    return <p className="text-sm text-slate-300 whitespace-pre-wrap">{value}</p>
  }

  // Number
  if (typeof value === 'number') {
    return <span className="text-lg font-bold text-white">{value}</span>
  }

  // Boolean
  if (typeof value === 'boolean') {
    return <span className="text-sm text-slate-300">{value ? 'Yes' : 'No'}</span>
  }

  // Ranked list: [{category/name/feature: str, count: n}, ...]
  if (isRankedList(value)) {
    return <RankedList items={value} />
  }

  // String array
  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'string') {
    return <StringList items={value as string[]} asQuotes={QUOTE_KEYS.has(fieldKey)} />
  }

  // Array of objects (e.g. displacement_map, category_insights, vendor_scorecards)
  if (
    Array.isArray(value) &&
    value.length > 0 &&
    typeof value[0] === 'object' &&
    value[0] !== null
  ) {
    return <DataTable rows={value as Record<string, unknown>[]} />
  }

  // Flat {key: number} object (e.g. by_buying_stage, seat_count_signals)
  if (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.values(value).every((v) => typeof v === 'number' || typeof v === 'string')
  ) {
    return <StatObject obj={value as Record<string, unknown>} />
  }

  // Nested {key: {subkey: value}} object (e.g. source_distribution)
  if (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.values(value).every(
      (v) => typeof v === 'object' && v !== null && !Array.isArray(v),
    )
  ) {
    const entries = Object.entries(value as Record<string, Record<string, unknown>>)
    // Convert to table rows
    const rows = entries.map(([k, v]) => ({ name: k, ...v }))
    return <DataTable rows={rows} />
  }

  // Fallback: formatted JSON
  return (
    <pre className="text-xs text-slate-400 bg-slate-800/50 rounded p-3 overflow-x-auto">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function ReportDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const { data: report, loading, error, refresh, refreshing } = useApiData<ReportDetailType>(
    () => {
      if (!id) return Promise.reject(new Error('Missing report ID'))
      return fetchReport(id)
    },
    [id],
  )

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />
  if (!report) return <PageError error={new Error('Report not found')} />

  const badgeColor = REPORT_TYPE_COLORS[report.report_type] ?? 'bg-slate-500/20 text-slate-400'
  const title = ['vendor_comparison', 'account_comparison'].includes(report.report_type) && report.vendor_filter && report.category_filter
    ? `${report.vendor_filter} vs ${report.category_filter}`
    : (report.vendor_filter ?? report.report_type.replace(/_/g, ' '))

  // intelligence_data can be an object (keyed fields) or an array (vendor/edge rows)
  const rawIntel = report.intelligence_data
  const intelIsArray = Array.isArray(rawIntel)
  const intel = intelIsArray ? {} : (rawIntel ?? {})
  const scalarEntries = Object.entries(intel).filter(([k]) => SCALAR_KEYS.has(k))
  const richEntries = Object.entries(intel).filter(([k, v]) => {
    if (SCALAR_KEYS.has(k)) return false
    // Skip duplicate of top-level executive_summary
    if (k === 'executive_summary') return false
    // Skip empty arrays / empty strings / null
    if (v === null || v === undefined) return false
    if (typeof v === 'string' && v.trim() === '') return false
    if (Array.isArray(v) && v.length === 0) return false
    return true
  })

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate('/reports')}
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Reports
        </button>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
        </button>
      </div>

      <div>
        <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mb-2', badgeColor)}>
          {report.report_type.replace(/_/g, ' ')}
        </span>
        <h1 className="text-2xl font-bold text-white">
          {title}
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          {report.report_date ?? report.created_at}
          {report.llm_model && ` | Model: ${report.llm_model}`}
        </p>
      </div>

      {report.executive_summary && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-2">Executive Summary</h3>
          <p className="text-sm text-slate-300 whitespace-pre-wrap">{report.executive_summary}</p>
        </div>
      )}

      {/* Challenger brief — dedicated renderer */}
      {report.report_type === 'challenger_brief' && !intelIsArray && (
        <ChallengerBriefDetail data={intel as AnyObj} />
      )}

      {/* Accounts in motion — dedicated renderer */}
      {report.report_type === 'accounts_in_motion' && !intelIsArray && (
        <AccountsInMotionDetail data={intel as AnyObj} />
      )}

      {/* Generic rendering for all other report types */}
      {!['challenger_brief', 'accounts_in_motion'].includes(report.report_type) && (
        <>
          {/* Array-based reports (scorecard, displacement, category overview, churn feed) */}
          {intelIsArray && (rawIntel as Record<string, unknown>[]).length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">
                {report.report_type.replace(/_/g, ' ')} ({(rawIntel as unknown[]).length} items)
              </h3>
              <DataTable rows={rawIntel as Record<string, unknown>[]} />
            </div>
          )}

          {/* Scalar stats row */}
          {scalarEntries.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {scalarEntries.map(([key, value]) => (
                <div key={key} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
                  <p className="text-xs text-slate-400 mb-1">{key.replace(/_/g, ' ')}</p>
                  <p className="text-xl font-bold text-white">{String(value)}</p>
                </div>
              ))}
            </div>
          )}

          {/* Rich intelligence fields */}
          {richEntries.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {richEntries.map(([key, value]) => (
                <div key={key} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <h4 className="text-xs font-medium text-cyan-400 uppercase tracking-wider mb-3">
                    {humanLabel(key)}
                  </h4>
                  <IntelValue fieldKey={key} value={value} />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {report.data_density && Object.keys(report.data_density).length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Data Density</h3>
          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
            {Object.entries(report.data_density).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <dt className="text-slate-400">{key.replace(/_/g, ' ')}</dt>
                <dd className="text-white">{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  )
}
