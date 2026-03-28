import { clsx } from 'clsx'
import { MessageSquareQuote, Shield, Swords, Target, TrendingDown, Zap } from 'lucide-react'
import type { ReactNode } from 'react'
import ArchetypeBadge from '@/components/ArchetypeBadge'
import { StructuredReportData } from '@/components/report-renderers/StructuredReportData'
import { normalizeReportObject, normalizeUnknown } from '@/lib/reportNormalization'
import {
  toBattleCardViewModel,
  toAccountsInMotionViewModel,
  toChallengerBriefViewModel,
  toComparisonReportViewModel,
  toVendorScorecards,
  toWeeklyChurnFeedItems,
} from '@/lib/reportViewModels'
import type {
  AccountsInMotionAccountViewModel,
  AccountsInMotionViewModel,
  BattleCardViewModel,
  ChallengerBriefViewModel,
  ChallengerTargetAccountViewModel,
  ComparisonReportViewModel,
  CompetitorDifferentiatorViewModel,
  FeatureGapViewModel,
  ObjectionHandlerViewModel,
  RecommendedPlayViewModel,
  TalkTrackViewModel,
  VendorScorecardViewModel,
  WeeklyChurnFeedItemViewModel,
  WeaknessAnalysisItemViewModel,
} from '@/lib/types/reportViewModels'

export const SPECIALIZED_REPORT_TYPES = [
  'challenger_brief',
  'accounts_in_motion',
  'battle_card',
  'vendor_comparison',
  'account_comparison',
  'weekly_churn_feed',
  'vendor_scorecard',
  'displacement_report',
  'category_overview',
  'vendor_deep_dive',
] as const

export function isSpecializedReportType(reportType: string): boolean {
  return SPECIALIZED_REPORT_TYPES.includes(reportType as (typeof SPECIALIZED_REPORT_TYPES)[number])
}

function SectionCard({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 min-w-0 overflow-hidden [overflow-wrap:anywhere]">
      <h3 className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-3 min-w-0 break-words">
        {icon}
        {title}
      </h3>
      {children}
    </div>
  )
}

function MetricRow({ label, value, color }: { label: string; value: string | number | null | undefined; color?: string }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3 text-sm items-start">
      <span className="text-slate-400 min-w-0 break-words">{label}</span>
      <span className={clsx('min-w-0 break-words text-right', color ?? 'text-white')}>{String(value)}</span>
    </div>
  )
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== 'number') return '--'
  return `${(value * 100).toFixed(1)}%`
}

function riskColor(level: string | undefined): string {
  const colors: Record<string, string> = {
    critical: 'text-red-400',
    high: 'text-orange-400',
    medium: 'text-yellow-400',
    low: 'text-green-400',
  }
  return colors[level ?? ''] ?? 'text-slate-300'
}

function ChallengerBriefDetail({ data }: { data: ChallengerBriefViewModel }) {
  const disp = data.displacement_summary
  const inc = data.incumbent_profile
  const adv = data.challenger_advantage
  const h2h = data.head_to_head
  const targets = data.target_accounts
  const playbook = data.sales_playbook
  const integ = data.integration_comparison
  const sources = data.data_sources

  return (
    <div className="space-y-6 min-w-0 [overflow-wrap:anywhere]">
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(sources).map(([key, value]) => (
          <span
            key={key}
            className={clsx(
              'px-2 py-0.5 rounded text-xs font-medium',
              value ? 'bg-cyan-900/40 text-cyan-300' : 'bg-slate-800 text-slate-600',
            )}
          >
            {key.replace(/_/g, ' ')}
          </span>
        ))}
      </div>

      <SectionCard title="Displacement Signal" icon={<TrendingDown className="h-4 w-4 text-red-400" />}>
        <div className="space-y-1">
          <MetricRow label="Total Mentions" value={disp.total_mentions} color={Number(disp.total_mentions) >= 50 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />
          <MetricRow label="Signal Strength" value={disp.signal_strength} />
          <MetricRow label="Confidence" value={typeof disp.confidence_score === 'number' ? formatPercent(disp.confidence_score) : disp.confidence_score} />
          <MetricRow label="Primary Driver" value={disp.primary_driver} />
        </div>
        {disp.source_distribution && typeof disp.source_distribution === 'object' && Object.keys(disp.source_distribution).length > 0 && (
          <p className="text-xs text-slate-500 mt-2">
            Sources: {Object.entries(disp.source_distribution as Record<string, number>)
              .sort(([, a], [, b]) => b - a)
              .map(([src, count]) => `${src}: ${count}`)
              .join(', ')}
          </p>
        )}
        {disp.key_quote && (
          <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 mt-2 break-words whitespace-pre-wrap">
            "{disp.key_quote}"
          </blockquote>
        )}
      </SectionCard>

      {Object.keys(inc).length > 0 && (
        <SectionCard title={`Incumbent: ${data.incumbent}`} icon={<Shield className="h-4 w-4 text-red-400" />}>
          <div className="space-y-1">
            {inc.archetype && (
              <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 items-start text-sm">
                <span className="text-slate-400 min-w-0 break-words">Archetype</span>
                <div className="min-w-0">
                  <ArchetypeBadge archetype={inc.archetype} confidence={inc.archetype_confidence} showConfidence />
                </div>
              </div>
            )}
            <MetricRow label="Risk Level" value={inc.risk_level} color={riskColor(inc.risk_level)} />
            {inc.churn_pressure_score != null && (
              <MetricRow label="Churn Pressure" value={`${Number(inc.churn_pressure_score).toFixed(0)}/100`} color={Number(inc.churn_pressure_score) >= 60 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />
            )}
            <MetricRow label="Price Complaint Rate" value={formatPercent(inc.price_complaint_rate)} />
            <MetricRow label="DM Churn Rate" value={formatPercent(inc.dm_churn_rate)} />
            <MetricRow label="Sentiment Trend" value={inc.sentiment_direction} />
          </div>

          {Array.isArray(inc.key_signals) && inc.key_signals.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Key Signals</p>
              <ul className="space-y-0.5">
                {inc.key_signals.slice(0, 5).map((signal: string, index: number) => (
                  <li key={index} className="text-xs text-slate-300 flex gap-2">
                    <span className="text-amber-400">-</span> {signal}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {Array.isArray(inc.top_weaknesses) && inc.top_weaknesses.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <p className="text-xs font-medium text-slate-400 mb-1">Top Weaknesses</p>
              <table className="w-full text-sm table-auto">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Weakness</th>
                    <th className="text-right text-xs text-slate-400 px-2 py-1 align-top break-words">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {inc.top_weaknesses.slice(0, 8).map((weakness: WeaknessAnalysisItemViewModel, index: number) => (
                    <tr key={index} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300 align-top break-words">{weakness.area ?? weakness.weakness ?? weakness.name ?? ''}</td>
                      <td className="px-2 py-1 text-slate-400 text-right align-top break-words">{weakness.count ?? weakness.evidence_count ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {Array.isArray(inc.top_pain_quotes) && inc.top_pain_quotes.length > 0 && (
            <div className="space-y-2 mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Customer Pain</p>
              {inc.top_pain_quotes.slice(0, 5).map((quote, index: number) => (
                <blockquote key={index} className="text-sm text-slate-300 italic border-l-2 border-red-500/50 pl-3">
                  "{typeof quote === 'string' ? quote : quote?.quote ?? ''}"
                  {quote?.source_site && <span className="text-xs text-slate-500 not-italic ml-2">({quote.source_site})</span>}
                </blockquote>
              ))}
            </div>
          )}
        </SectionCard>
      )}

      {(Array.isArray(adv.strengths) && adv.strengths.length > 0) || adv.profile_summary ? (
        <SectionCard title={`Challenger: ${data.challenger}`} icon={<Zap className="h-4 w-4 text-green-400" />}>
          {adv.profile_summary && <p className="text-sm text-slate-300 mb-3">{adv.profile_summary}</p>}

          {Array.isArray(adv.strengths) && adv.strengths.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm table-auto">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Strength</th>
                    <th className="text-right text-xs text-slate-400 px-2 py-1 align-top break-words">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {adv.strengths.slice(0, 8).map((strength, index: number) => (
                    <tr key={index} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300 align-top break-words">{strength.area ?? strength.name ?? ''}</td>
                      <td className="px-2 py-1 text-slate-400 text-right align-top break-words">{strength.evidence_count ?? strength.mentions ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {Array.isArray(adv.weakness_coverage) && adv.weakness_coverage.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <p className="text-xs font-medium text-slate-400 mb-1">Weakness Coverage</p>
              <table className="w-full text-sm table-auto">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Incumbent Weakness</th>
                    <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Match</th>
                  </tr>
                </thead>
                <tbody>
                  {adv.weakness_coverage.slice(0, 8).map((coverage, index: number) => (
                    <tr key={index} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300 align-top break-words">{coverage.incumbent_weakness ?? ''}</td>
                      <td className="px-2 py-1">
                        <span className={clsx('text-xs px-1.5 py-0.5 rounded', coverage.match_quality === 'strong' ? 'bg-green-500/15 text-green-300' : 'bg-amber-500/15 text-amber-300')}>
                          {coverage.match_quality}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {Array.isArray(adv.commonly_switched_from) && adv.commonly_switched_from.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Commonly Switched From</p>
              <div className="flex flex-wrap gap-1.5">
                {adv.commonly_switched_from.slice(0, 10).map((switchFrom: string, index: number) => (
                  <span key={index} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                    {switchFrom}
                  </span>
                ))}
              </div>
            </div>
          )}
        </SectionCard>
      ) : null}

      {(h2h.conclusion || h2h.winner) && (
        <SectionCard title="Head to Head" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
          <div className="space-y-1">
            {h2h.winner && <MetricRow label="Winner" value={h2h.winner} color="text-cyan-400 font-bold" />}
            {h2h.confidence != null && <MetricRow label="Confidence" value={formatPercent(h2h.confidence)} />}
            {h2h.durability && <MetricRow label="Durability" value={h2h.durability} />}
            {h2h.synthesized && <span className="text-[10px] text-slate-600">(synthesized from displacement data)</span>}
          </div>
          {h2h.conclusion && <p className="text-sm text-slate-300 mt-2">{h2h.conclusion}</p>}
          {Array.isArray(h2h.key_insights) && h2h.key_insights.length > 0 && (
            <ul className="space-y-1 mt-2">
              {h2h.key_insights.slice(0, 5).map((insight, index: number) => {
                const text = insight.insight ?? ''
                const evidence = insight.evidence ?? ''
                return (
                  <li key={index} className="text-xs text-slate-400 flex gap-2 min-w-0">
                    <span className="text-cyan-400 shrink-0">-</span>
                    <span className="min-w-0 break-words">{text}{evidence && <span className="text-slate-600 ml-1 break-all">({evidence})</span>}</span>
                  </li>
                )
              })}
            </ul>
          )}
        </SectionCard>
      )}

      {targets.length > 0 && (
        <SectionCard title={`Target Accounts (${data.total_target_accounts ?? targets.length} total, ${data.accounts_considering_challenger ?? 0} considering ${data.challenger})`} icon={<Target className="h-4 w-4 text-amber-400" />}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm table-auto">
              <thead>
                <tr className="border-b border-slate-700/50">
                  {['Company', 'Score', 'Stage', 'Urg', 'Industry', 'Chall?'].map((header) => (
                    <th key={header} className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {targets.slice(0, 15).map((target: ChallengerTargetAccountViewModel, index: number) => (
                  <tr key={index} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{target.company ?? ''}</td>
                    <td className="px-2 py-1 text-white font-medium">{target.opportunity_score ?? ''}</td>
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{(target.buying_stage ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{typeof target.urgency === 'number' ? target.urgency.toFixed(0) : ''}</td>
                    <td className="px-2 py-1 text-slate-400 align-top break-words">{(target.industry ?? '').slice(0, 20)}</td>
                    <td className="px-2 py-1">{target.considers_challenger ? <span className="text-green-400">Y</span> : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {Object.keys(playbook).length > 0 && (
        <SectionCard title="Sales Playbook" icon={<MessageSquareQuote className="h-4 w-4 text-cyan-400" />}>
          {Array.isArray(playbook.discovery_questions) && playbook.discovery_questions.length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Discovery Questions</p>
              <ol className="space-y-1.5 list-none">
                {playbook.discovery_questions.slice(0, 5).map((question: string, index: number) => (
                  <li key={index} className="flex gap-2 text-sm">
                    <span className="shrink-0 w-5 h-5 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-xs">{index + 1}</span>
                    <span className="text-slate-300">{question}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {Array.isArray(playbook.landmine_questions) && playbook.landmine_questions.length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Landmine Questions</p>
              <ol className="space-y-1.5 list-none">
                {playbook.landmine_questions.slice(0, 3).map((question: string, index: number) => (
                  <li key={index} className="flex gap-2 text-sm">
                    <span className="shrink-0 w-5 h-5 rounded-full bg-red-500/15 text-red-400 flex items-center justify-center text-xs">{index + 1}</span>
                    <span className="text-slate-300">{question}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {Array.isArray(playbook.objection_handlers) && playbook.objection_handlers.length > 0 && (
            <div className="mb-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Objection Handlers</p>
              <div className="space-y-2">
                {playbook.objection_handlers.map((handler: ObjectionHandlerViewModel, index: number) => (
                  <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                    {handler.objection && <p className="text-sm text-red-300 mb-1"><span className="text-xs text-slate-500 uppercase mr-1">Objection:</span>{handler.objection}</p>}
                    {handler.pivot && <p className="text-sm text-green-300"><span className="text-xs text-slate-500 uppercase mr-1">Pivot:</span>{handler.pivot}</p>}
                    {handler.proof_point && <p className="text-xs text-slate-400 mt-1">{handler.proof_point}</p>}
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
                  { key: 'opening' as const, label: 'Opening', color: 'border-cyan-500/50' },
                  { key: 'mid_call_pivot' as const, label: 'Mid-Call Pivot', color: 'border-amber-500/50' },
                  { key: 'closing' as const, label: 'Closing', color: 'border-green-500/50' },
                ].map(({ key, label, color }) => {
                  const value = (playbook.talk_track as TalkTrackViewModel)[key]
                  if (!value) return null
                  return (
                    <div key={key} className={`border-l-2 ${color} pl-3`}>
                      <p className="text-xs text-slate-500 uppercase mb-1">{label}</p>
                      <p className="text-sm text-slate-300">{String(value)}</p>
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
                {playbook.recommended_plays.map((play: RecommendedPlayViewModel, index: number) => (
                  <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                    <p className="text-sm text-white font-medium">{play.play ?? play.name ?? play.description ?? ''}</p>
                    {play.target_segment && <span className="text-xs text-amber-300">{play.target_segment}</span>}
                    {play.key_message && <p className="text-xs text-slate-400 mt-1">{play.key_message}</p>}
                    {play.timing && <p className="text-xs text-slate-500 mt-1">Timing: {play.timing}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </SectionCard>
      )}

      {((Array.isArray(integ.shared) && integ.shared.length > 0) || (Array.isArray(integ.challenger_exclusive) && integ.challenger_exclusive.length > 0)) && (
        <SectionCard title="Integration Comparison">
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
        </SectionCard>
      )}
    </div>
  )
}

function AccountsInMotionDetail({ data }: { data: AccountsInMotionViewModel }) {
  const pricing = data.pricing_pressure
  const gaps = data.feature_gaps
  const xvc = data.cross_vendor_context
  const accounts = data.accounts

  return (
    <div className="space-y-6 min-w-0 [overflow-wrap:anywhere]">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Total Accounts</p>
          <p className="text-2xl font-bold text-cyan-400">{data.total_accounts_in_motion ?? accounts.length}</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Archetype</p>
          {data.archetype ? <ArchetypeBadge archetype={data.archetype} confidence={data.archetype_confidence} showConfidence /> : <p className="text-sm text-slate-500">--</p>}
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Price Complaint Rate</p>
          <p className="text-xl font-bold text-white">{pricing.price_complaint_rate != null ? formatPercent(pricing.price_complaint_rate) : '--'}</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Top Destination</p>
          <p className="text-sm font-medium text-white">{xvc.top_destination ?? '--'}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {(pricing.price_complaint_rate != null || pricing.price_increase_rate != null) && (
          <SectionCard title="Pricing Pressure" icon={<TrendingDown className="h-4 w-4 text-amber-400" />}>
            <div className="space-y-1">
              <MetricRow label="Price Complaint Rate" value={formatPercent(pricing.price_complaint_rate)} />
              <MetricRow label="Price Increase Rate" value={formatPercent(pricing.price_increase_rate)} />
              {pricing.avg_seat_count != null && <MetricRow label="Avg Seat Count" value={Math.round(pricing.avg_seat_count)} />}
            </div>
          </SectionCard>
        )}

        {gaps.length > 0 && (
          <SectionCard title="Top Feature Gaps">
            <div className="overflow-x-auto">
              <table className="w-full text-sm table-auto">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Feature</th>
                    <th className="text-right text-xs text-slate-400 px-2 py-1 align-top break-words">Mentions</th>
                  </tr>
                </thead>
                <tbody>
                  {gaps.slice(0, 10).map((gap: FeatureGapViewModel, index: number) => (
                    <tr key={index} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300 align-top break-words">{gap.feature ?? ''}</td>
                      <td className="px-2 py-1 text-slate-400 text-right align-top break-words">{gap.mentions ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        )}
      </div>

      {(xvc.top_destination || xvc.battle_conclusion || xvc.market_regime) && (
        <SectionCard title="Competitive Context" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
          <div className="space-y-1">
            <MetricRow label="Top Destination" value={xvc.top_destination} color="text-cyan-400" />
            <MetricRow label="Market Regime" value={(xvc.market_regime ?? '').replace(/_/g, ' ')} />
          </div>
          {xvc.battle_conclusion && <p className="text-sm text-slate-300 mt-2">{xvc.battle_conclusion}</p>}
        </SectionCard>
      )}

      {accounts.length > 0 && (
        <SectionCard title={`Prospecting List (${accounts.length} accounts)`} icon={<Target className="h-4 w-4 text-amber-400" />}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm table-auto">
              <thead>
                <tr className="border-b border-slate-700/50">
                  {['Company', 'Score', 'Stage', 'Urg', 'Pain', 'Industry', 'Domain', 'Alts'].map((header) => (
                    <th key={header} className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {accounts.slice(0, 25).map((account: AccountsInMotionAccountViewModel, index: number) => (
                  <tr key={index} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{(account.company ?? '').slice(0, 24)}</td>
                    <td className="px-2 py-1">
                      <span className={clsx('font-medium', Number(account.opportunity_score) >= 50 ? 'text-green-400' : Number(account.opportunity_score) >= 30 ? 'text-amber-400' : 'text-slate-300')}>
                        {account.opportunity_score ?? ''}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{(account.buying_stage ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{typeof account.urgency === 'number' ? account.urgency.toFixed(0) : ''}</td>
                    <td className="px-2 py-1 text-slate-400 align-top break-words">{(account.pain_category ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-400 align-top break-words">{(account.industry ?? '').slice(0, 18)}</td>
                    <td className="px-2 py-1 text-slate-400 align-top break-words">{(account.domain ?? '').slice(0, 22)}</td>
                    <td className="px-2 py-1 text-xs text-slate-500 align-top break-words">{Array.isArray(account.alternatives_considering) ? account.alternatives_considering.slice(0, 2).join(', ') : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {accounts.filter((account) => account.top_quote).length > 0 && (
            <div className="mt-4">
              <p className="text-xs font-medium text-slate-400 mb-2">Key Quotes</p>
              <div className="space-y-2">
                {accounts.filter((account) => account.top_quote).slice(0, 5).map((account, index) => (
                  <div key={index}>
                    <span className="text-[10px] text-slate-500">{account.company}{account.urgency ? ` (urgency: ${account.urgency})` : ''}</span>
                    <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 break-words whitespace-pre-wrap">
                      "{account.top_quote}"
                    </blockquote>
                  </div>
                ))}
              </div>
            </div>
          )}
        </SectionCard>
      )}
    </div>
  )
}

function BattleCardDetail({ data, rawData }: { data: BattleCardViewModel; rawData: Record<string, unknown> }) {
  const weaknesses = data.weakness_analysis.length > 0 ? data.weakness_analysis : data.vendor_weaknesses
  const qualityClass = data.quality_status === 'sales_ready'
    ? 'bg-emerald-500/15 text-emerald-300'
    : data.quality_status === 'needs_review'
      ? 'bg-amber-500/15 text-amber-300'
      : data.quality_status === 'deterministic_fallback'
        ? 'bg-rose-500/15 text-rose-300'
        : 'bg-slate-500/15 text-slate-300'
  const qualityLabel = data.quality_status
    ? data.quality_status.replace(/_/g, ' ')
    : null
  const skipKeys = [
    'vendor', 'category', 'churn_pressure_score', 'total_reviews', 'confidence',
    'archetype', 'archetype_risk_level', 'archetype_key_signals',
    'vendor_weaknesses', 'weakness_analysis', 'customer_pain_quotes',
    'competitor_differentiators', 'cross_vendor_battles', 'competitive_landscape',
    'resource_asymmetry', 'category_council', 'objection_handlers', 'talk_track',
    'recommended_plays', 'active_evaluation_deadlines', 'source_distribution',
    'llm_render_status', 'quality_status', 'quality_score', 'battle_card_quality',
  ]

  return (
    <div className="space-y-6 min-w-0 [overflow-wrap:anywhere]">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Churn Pressure</p>
          <p className="text-2xl font-bold text-red-400">{data.churn_pressure_score != null ? Math.round(data.churn_pressure_score) : '--'}</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Reviews</p>
          <p className="text-2xl font-bold text-white">{data.total_reviews ?? '--'}</p>
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Archetype</p>
          {data.archetype ? <ArchetypeBadge archetype={data.archetype} showConfidence /> : <p className="text-sm text-slate-500">--</p>}
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
          <p className="text-xs text-slate-400 mb-1">Render</p>
          <p className="text-sm font-medium text-white">{data.llm_render_status ?? 'unknown'}</p>
        </div>
      </div>
      {(qualityLabel || data.quality_score != null) && (
        <div className="flex flex-wrap items-center gap-2">
          {qualityLabel && (
            <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${qualityClass}`}>
              {qualityLabel}
            </span>
          )}
          {data.quality_score != null && (
            <span className="text-xs text-slate-400">quality score: <span className="text-white font-medium">{Math.round(data.quality_score)}</span></span>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {weaknesses.length > 0 && (
          <SectionCard title="Weakness Analysis" icon={<TrendingDown className="h-4 w-4 text-red-400" />}>
            <div className="space-y-3">
              {weaknesses.slice(0, 5).map((weakness: WeaknessAnalysisItemViewModel, index: number) => (
                <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-sm font-medium text-white">{weakness.weakness ?? weakness.area ?? weakness.name ?? ''}</p>
                  {weakness.evidence && <p className="text-xs text-slate-400 mt-1">{weakness.evidence}</p>}
                  {weakness.customer_quote && <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 mt-2 break-words whitespace-pre-wrap">"{weakness.customer_quote}"</blockquote>}
                  {weakness.winning_position && <p className="text-xs text-cyan-300 mt-2">{weakness.winning_position}</p>}
                </div>
              ))}
            </div>
          </SectionCard>
        )}

        {data.competitive_landscape && (
          <SectionCard title="Competitive Landscape" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
            {data.competitive_landscape.vulnerability_window && (
              <p className="text-sm text-slate-300 mb-3">{data.competitive_landscape.vulnerability_window}</p>
            )}
            {data.competitive_landscape.top_alternatives.length > 0 && (
              <div className="mb-3">
                <p className="text-xs font-medium text-slate-400 mb-1">Top Alternatives</p>
                <div className="flex flex-wrap gap-1.5">
                  {data.competitive_landscape.top_alternatives.map((item, index) => (
                    <span key={index} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">{item}</span>
                  ))}
                </div>
              </div>
            )}
            {data.competitive_landscape.displacement_triggers.length > 0 && (
              <div>
                <p className="text-xs font-medium text-slate-400 mb-1">Displacement Triggers</p>
                <ul className="space-y-1">
                  {data.competitive_landscape.displacement_triggers.slice(0, 4).map((trigger, index) => (
                    <li key={index} className="text-sm text-slate-300 flex gap-2 min-w-0">
                      <span className="text-cyan-400 shrink-0">-</span>
                      <span className="min-w-0 break-words">{trigger}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </SectionCard>
        )}
      </div>

      {(data.customer_pain_quotes.length > 0 || data.competitor_differentiators.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {data.customer_pain_quotes.length > 0 && (
            <SectionCard title="Customer Pain Quotes" icon={<MessageSquareQuote className="h-4 w-4 text-amber-400" />}>
              <div className="space-y-3">
                {data.customer_pain_quotes.slice(0, 5).map((quote, index) => (
                  <div key={index}>
                    <blockquote className="text-sm text-slate-300 italic border-l-2 border-amber-500/50 pl-3 break-words whitespace-pre-wrap">"{quote.quote}"</blockquote>
                    <div className="flex flex-wrap gap-2 mt-1 text-xs text-slate-500">
                      {quote.company && <span>{quote.company}</span>}
                      {quote.role && <span>{quote.role}</span>}
                      {quote.pain_category && <span>{quote.pain_category}</span>}
                      {quote.urgency != null && <span className="text-amber-400">urgency {quote.urgency}/10</span>}
                    </div>
                  </div>
                ))}
              </div>
            </SectionCard>
          )}

          {data.competitor_differentiators.length > 0 && (
            <SectionCard title="Competitor Differentiators" icon={<Zap className="h-4 w-4 text-green-400" />}>
              <div className="overflow-x-auto">
                <table className="w-full text-sm table-auto">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Competitor</th>
                      <th className="text-right text-xs text-slate-400 px-2 py-1 align-top break-words">Mentions</th>
                      <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Driver</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.competitor_differentiators.slice(0, 8).map((item: CompetitorDifferentiatorViewModel, index: number) => (
                      <tr key={index} className="border-b border-slate-800/50">
                        <td className="px-2 py-1 text-slate-300 align-top break-words">{item.competitor ?? ''}</td>
                        <td className="px-2 py-1 text-right text-slate-400 align-top break-words">{item.mentions ?? item.count ?? ''}</td>
                        <td className="px-2 py-1 text-slate-400 align-top break-words">{item.primary_driver ?? item.solves_weakness ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          )}
        </div>
      )}

      {(data.cross_vendor_battles.length > 0 || data.resource_asymmetry || data.category_council) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {data.cross_vendor_battles.length > 0 && (
            <SectionCard title="Cross-Vendor Battles" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
              <div className="space-y-3">
                {data.cross_vendor_battles.slice(0, 4).map((battle, index) => (
                  <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                    <div className="flex flex-wrap gap-2 text-xs mb-2">
                      {battle.opponent && <span className="text-white font-medium">vs {battle.opponent}</span>}
                      {battle.winner && <span className="px-2 py-0.5 bg-emerald-500/15 text-emerald-300 rounded">winner: {battle.winner}</span>}
                      {battle.loser && <span className="px-2 py-0.5 bg-red-500/15 text-red-300 rounded">loser: {battle.loser}</span>}
                    </div>
                    {battle.conclusion && <p className="text-sm text-slate-300">{battle.conclusion}</p>}
                  </div>
                ))}
              </div>
            </SectionCard>
          )}

          {(data.resource_asymmetry || data.category_council) && (
            <SectionCard title="Market Context" icon={<Shield className="h-4 w-4 text-cyan-400" />}>
              {data.resource_asymmetry?.conclusion && <p className="text-sm text-slate-300 mb-3">{data.resource_asymmetry.conclusion}</p>}
              {data.resource_asymmetry?.resource_advantage && <MetricRow label="Resource Advantage" value={data.resource_asymmetry.resource_advantage} />}
              {data.category_council?.conclusion && <p className="text-sm text-slate-300 mt-3">{data.category_council.conclusion}</p>}
              {data.category_council?.key_insights.length ? (
                <ul className="space-y-1 mt-3">
                  {data.category_council.key_insights.slice(0, 4).map((insight, index) => (
                    <li key={index} className="text-xs text-slate-400 flex gap-2 min-w-0">
                      <span className="text-cyan-400 shrink-0">-</span>
                      <span className="min-w-0 break-all">{insight.insight}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </SectionCard>
          )}
        </div>
      )}

      {(data.active_evaluation_deadlines.length > 0 || data.recommended_plays.length > 0 || data.talk_track) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {data.active_evaluation_deadlines.length > 0 && (
            <SectionCard title="Active Evaluation Deadlines" icon={<Target className="h-4 w-4 text-red-400" />}>
              <div className="overflow-x-auto">
                <table className="w-full text-sm table-auto">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Company</th>
                      <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Timeline</th>
                      <th className="text-right text-xs text-slate-400 px-2 py-1 align-top break-words">Urgency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.active_evaluation_deadlines.slice(0, 8).map((item, index) => (
                      <tr key={index} className="border-b border-slate-800/50">
                        <td className="px-2 py-1 text-slate-300 align-top break-words">{item.company ?? ''}</td>
                        <td className="px-2 py-1 text-slate-400 align-top break-words">{item.decision_timeline ?? item.evaluation_deadline ?? item.contract_end ?? ''}</td>
                        <td className="px-2 py-1 text-right text-slate-400 align-top break-words">{item.urgency ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          )}

          {(data.talk_track || data.recommended_plays.length > 0) && (
            <SectionCard title="Talk Track and Plays" icon={<MessageSquareQuote className="h-4 w-4 text-cyan-400" />}>
              {data.talk_track && (
                <div className="space-y-3 mb-4">
                  {[
                    { key: 'opening' as const, label: 'Opening' },
                    { key: 'mid_call_pivot' as const, label: 'Mid-Call Pivot' },
                    { key: 'closing' as const, label: 'Closing' },
                  ].map(({ key, label }) => {
                    const value = data.talk_track?.[key]
                    if (!value) return null
                    return <div key={key}><p className="text-xs text-slate-500 uppercase mb-1">{label}</p><p className="text-sm text-slate-300 break-words">{value}</p></div>
                  })}
                </div>
              )}
              {data.recommended_plays.length > 0 && (
                <div className="space-y-2">
                  {data.recommended_plays.slice(0, 4).map((play: RecommendedPlayViewModel, index: number) => (
                    <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                      <p className="text-sm text-white font-medium">{play.play ?? play.name ?? play.description ?? ''}</p>
                      {play.target_segment && <p className="text-xs text-amber-300 mt-1">{play.target_segment}</p>}
                      {play.key_message && <p className="text-xs text-slate-400 mt-1">{play.key_message}</p>}
                    </div>
                  ))}
                </div>
              )}
            </SectionCard>
          )}
        </div>
      )}

      {Object.keys(data.source_distribution).length > 0 && (
        <SectionCard title="Source Distribution">
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(data.source_distribution).sort(([, a], [, b]) => b - a).map(([source, count]) => (
              <span key={source} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">{source}: {count}</span>
            ))}
          </div>
        </SectionCard>
      )}

      <StructuredReportData data={rawData} skipKeys={skipKeys} className="xl:grid-cols-1" />
    </div>
  )
}

function ComparisonReportDetail({ data, rawData }: { data: ComparisonReportViewModel; rawData: Record<string, unknown> }) {
  const skipKeys = [
    'primary_vendor', 'comparison_vendor', 'primary_company', 'comparison_company',
    'report_date', 'window_days', 'primary_metrics', 'comparison_metrics',
    'primary_company_metrics', 'comparison_company_metrics', 'primary_top_pains',
    'comparison_top_pains', 'primary_quote_highlights', 'comparison_quote_highlights',
    'primary_strengths', 'primary_weaknesses', 'comparison_strengths',
    'comparison_weaknesses', 'primary_switching_triggers', 'comparison_switching_triggers',
    'shared_pain_categories', 'shared_alternatives', 'shared_vendors',
    'direct_displacement', 'trend_analysis', 'urgency_gap', 'vendor_archetypes',
  ]

  const metricRows = [
    ['Reviews', data.primary_metrics.signal_count, data.comparison_metrics.signal_count],
    ['Signal Density', data.primary_metrics.churn_signal_density, data.comparison_metrics.churn_signal_density],
    ['Avg Urgency', data.primary_metrics.avg_urgency_score, data.comparison_metrics.avg_urgency_score],
    ['Positive Review %', data.primary_metrics.positive_review_pct, data.comparison_metrics.positive_review_pct],
    ['Recommend Ratio', data.primary_metrics.recommend_ratio, data.comparison_metrics.recommend_ratio],
    ['Churn Intent', data.primary_metrics.churn_intent_count, data.comparison_metrics.churn_intent_count],
  ]

  return (
    <div className="space-y-6 min-w-0 [overflow-wrap:anywhere]">
      <SectionCard title="Side-by-Side Metrics" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm table-auto">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">Metric</th>
                <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">{data.primary_name ?? 'Primary'}</th>
                <th className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">{data.comparison_name ?? 'Comparison'}</th>
              </tr>
            </thead>
            <tbody>
              {metricRows.map(([label, primary, comparison]) => (
                <tr key={String(label)} className="border-b border-slate-800/50">
                  <td className="px-2 py-1 text-slate-400 align-top break-words">{label}</td>
                  <td className="px-2 py-1 text-slate-300 align-top break-words">{primary ?? '--'}</td>
                  <td className="px-2 py-1 text-slate-300 align-top break-words">{comparison ?? '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SectionCard title={`${data.primary_name ?? 'Primary'} Snapshot`}>
          {data.primary_top_pains.length > 0 && <MetricRow label="Top Pain" value={data.primary_top_pains[0]?.feature ?? data.primary_top_pains[0]?.category} />}
          {data.primary_quote_highlights.slice(0, 2).map((quote, index) => (
            <blockquote key={index} className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 mt-2">"{quote}"</blockquote>
          ))}
        </SectionCard>

        <SectionCard title={`${data.comparison_name ?? 'Comparison'} Snapshot`}>
          {data.comparison_top_pains.length > 0 && <MetricRow label="Top Pain" value={data.comparison_top_pains[0]?.feature ?? data.comparison_top_pains[0]?.category} />}
          {data.comparison_quote_highlights.slice(0, 2).map((quote, index) => (
            <blockquote key={index} className="text-sm text-slate-300 italic border-l-2 border-amber-500/50 pl-3 mt-2">"{quote}"</blockquote>
          ))}
        </SectionCard>
      </div>

      {(data.shared_pain_categories.length > 0 || data.shared_alternatives.length > 0 || data.shared_vendors.length > 0) && (
        <SectionCard title="Shared Context" icon={<Shield className="h-4 w-4 text-amber-400" />}>
          {data.shared_pain_categories.length > 0 && <MetricRow label="Shared Pain Categories" value={data.shared_pain_categories.join(', ')} />}
          {data.shared_alternatives.length > 0 && <MetricRow label="Shared Alternatives" value={data.shared_alternatives.join(', ')} />}
          {data.shared_vendors.length > 0 && <MetricRow label="Shared Vendors" value={data.shared_vendors.join(', ')} />}
          {data.urgency_gap != null && <MetricRow label="Urgency Gap" value={data.urgency_gap} color="text-cyan-400" />}
        </SectionCard>
      )}

      {(data.direct_displacement.length > 0 || data.trend_analysis || Object.keys(data.vendor_archetypes).length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {data.direct_displacement.length > 0 && (
            <SectionCard title="Direct Displacement" icon={<TrendingDown className="h-4 w-4 text-red-400" />}>
              <div className="space-y-3">
                {data.direct_displacement.slice(0, 4).map((flow, index) => (
                  <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                    <p className="text-sm text-slate-300">{flow.name ?? ''}</p>
                    {flow.count != null && <p className="text-xs text-slate-500 mt-1">Mentions: {flow.count}</p>}
                    {flow.companies.length > 0 && <p className="text-xs text-slate-500 mt-1">Companies: {flow.companies.slice(0, 3).join(', ')}</p>}
                  </div>
                ))}
              </div>
            </SectionCard>
          )}

          {(data.trend_analysis || Object.keys(data.vendor_archetypes).length > 0) && (
            <SectionCard title="Trend and Archetypes" icon={<Zap className="h-4 w-4 text-cyan-400" />}>
              {data.trend_analysis?.prior_report_date && <MetricRow label="Prior Report" value={data.trend_analysis.prior_report_date} />}
              {data.trend_analysis?.primary_churn_density_change != null && <MetricRow label="Primary Density Change" value={data.trend_analysis.primary_churn_density_change} />}
              {data.trend_analysis?.comparison_churn_density_change != null && <MetricRow label="Comparison Density Change" value={data.trend_analysis.comparison_churn_density_change} />}
              {Object.keys(data.vendor_archetypes).length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {Object.entries(data.vendor_archetypes).map(([vendor, detail]) => (
                    <span key={vendor} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                      {vendor}: {detail.archetype ?? 'unknown'}
                    </span>
                  ))}
                </div>
              )}
            </SectionCard>
          )}
        </div>
      )}

      <StructuredReportData data={rawData} skipKeys={skipKeys} />
    </div>
  )
}

function WeeklyChurnFeedDetail({ items }: { items: WeeklyChurnFeedItemViewModel[] }) {
  return (
    <div className="space-y-3 min-w-0 [overflow-wrap:anywhere]">
      {items.map((item, index) => {
        const maxPain = Number(item.pain_breakdown[0]?.count ?? item.pain_breakdown[0]?.mentions ?? 1)
        return (
          <div
            key={`${item.vendor ?? 'feed'}-${index}`}
            className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 space-y-2.5"
          >
            {/* Header row */}
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
                <TrendingDown className="h-4 w-4 text-red-400 shrink-0" />
                {item.vendor ?? 'Vendor'}
              </h3>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                {item.category && <span className="text-slate-400 whitespace-nowrap">{item.category}</span>}
                {item.churn_pressure_score != null && (
                  <span className="text-red-400 font-bold whitespace-nowrap">{Math.round(item.churn_pressure_score)} pressure</span>
                )}
                {item.avg_urgency != null && <span className="text-amber-400 whitespace-nowrap">{item.avg_urgency} urgency</span>}
                {item.total_reviews != null && <span className="text-slate-500 whitespace-nowrap">{item.total_reviews} reviews</span>}
              </div>
            </div>

            {/* Action recommendation */}
            {item.action_recommendation && (
              <p className="text-xs text-cyan-300 leading-relaxed">{item.action_recommendation}</p>
            )}

            {/* Pain + Displacement side by side */}
            {(item.pain_breakdown.length > 0 || item.top_displacement_targets.length > 0) && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {item.pain_breakdown.length > 0 && (
                  <div>
                    <p className="text-xs text-slate-500 mb-1.5">Pain Breakdown</p>
                    <div className="space-y-1.5">
                      {item.pain_breakdown.slice(0, 5).map((pain, j) => {
                        const label = (pain.category ?? pain.feature ?? '').replace(/_/g, ' ')
                        const count = Number(pain.count ?? pain.mentions ?? 0)
                        const pct = maxPain > 0 ? Math.min((count / maxPain) * 100, 100) : 0
                        return (
                          <div key={j} className="flex items-center gap-2 text-xs">
                            <span className="w-28 text-slate-300 shrink-0 capitalize truncate">{label}</span>
                            <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                              <div className="h-1.5 rounded-full bg-red-500/60" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-slate-500 shrink-0 w-8 text-right">{count}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {item.top_displacement_targets.length > 0 && (
                  <div>
                    <p className="text-xs text-slate-500 mb-1.5">Displacement Targets</p>
                    <div className="space-y-1">
                      {item.top_displacement_targets.slice(0, 5).map((t: CompetitorDifferentiatorViewModel, j: number) => (
                        <div key={j} className="flex items-center justify-between text-xs">
                          <span className="text-green-400">{t.competitor ?? '—'}</span>
                          <span className="text-slate-500">{t.mentions ?? t.count ?? '—'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Key quote */}
            {item.key_quote && (
              <blockquote className="text-xs text-slate-400 italic border-l-2 border-cyan-500/30 pl-2 break-words">
                "{item.key_quote}"
              </blockquote>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ---- Category Overview ----

interface CategoryOverviewItem {
  category?: string | null
  highest_churn_risk?: string | null
  emerging_challenger?: string | null
  dominant_pain?: string | null
  market_shift_signal?: string | null
  market_loser?: string | null
  market_winner?: string | null
  market_regime?: string | null
  market_insights?: unknown
  vendor_rankings?: Array<{ vendor?: string | null; churn_pressure_score?: number | null; churn_signal_density?: number | null; risk_level?: string | null }>
  top_feature_gaps?: Array<{ feature?: string | null; mentions?: number | null }>
  case_studies?: Array<{ vendor?: string | null; quote?: string | null; company?: string | null }>
}

function extractInsightText(item: unknown): string {
  if (typeof item === 'string') return item
  if (item && typeof item === 'object') {
    const r = item as Record<string, unknown>
    return String(r.insight ?? r.text ?? r.summary ?? r.signal ?? '')
  }
  return ''
}

function CategoryOverviewDetail({ items }: { items: CategoryOverviewItem[] }) {
  if (items.length === 0) return <p className="text-slate-500 text-sm">No categories found.</p>

  return (
    <div className="space-y-4 min-w-0">
      {/* Summary table */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 overflow-x-auto">
        <table className="w-full text-xs min-w-[600px]">
          <thead>
            <tr className="text-slate-500 border-b border-slate-700/50 text-left">
              <th className="pb-2 font-normal pr-4">Category</th>
              <th className="pb-2 font-normal pr-3">Top Pain</th>
              <th className="pb-2 font-normal pr-3">At Risk</th>
              <th className="pb-2 font-normal pr-3">Challenger</th>
              <th className="pb-2 font-normal pr-3">Regime</th>
              <th className="pb-2 font-normal">Signal</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {items.map((item, i) => (
              <tr key={i} className="hover:bg-slate-800/20 transition-colors">
                <td className="py-1.5 pr-4 text-slate-200 font-medium whitespace-nowrap">{item.category ?? '—'}</td>
                <td className="py-1.5 pr-3">
                  {item.dominant_pain
                    ? <DriverBadge driver={item.dominant_pain} />
                    : <span className="text-slate-500">—</span>}
                </td>
                <td className="py-1.5 pr-3 text-red-400 whitespace-nowrap">{item.highest_churn_risk ?? item.market_loser ?? '—'}</td>
                <td className="py-1.5 pr-3 text-green-400 whitespace-nowrap">{item.emerging_challenger ?? item.market_winner ?? '—'}</td>
                <td className="py-1.5 pr-3 text-slate-400 whitespace-nowrap capitalize">
                  {typeof item.market_regime === 'string' ? item.market_regime.replace(/_/g, ' ') : '—'}
                </td>
                <td className="py-1.5 text-slate-400 max-w-[18rem]">
                  <span className="line-clamp-2">{item.market_shift_signal ?? '—'}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-category detail panels */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {items.map((item, i) => {
          const rankings = item.vendor_rankings ?? []
          const gaps = item.top_feature_gaps ?? []
          const study = item.case_studies?.[0]
          const insights = Array.isArray(item.market_insights) ? item.market_insights : []

          return (
            <div key={i} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-3 space-y-2.5">
              <h3 className="text-sm font-semibold text-slate-200">{item.category ?? 'Category'}</h3>

              {rankings.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-1">Vendor Risk</p>
                  <div className="space-y-0.5">
                    {rankings.slice(0, 5).map((vr, j) => (
                      <div key={j} className="flex items-center justify-between text-xs">
                        <span className={riskColor(vr.risk_level ?? '')}>{vr.vendor ?? '—'}</span>
                        <span className="text-slate-600">{vr.churn_pressure_score ?? vr.churn_signal_density ?? '—'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {gaps.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-1">Top Gaps</p>
                  <div className="flex flex-wrap gap-1">
                    {gaps.slice(0, 4).map((g, j) => (
                      <span key={j} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-xs">
                        {g.feature ?? '?'}{g.mentions != null ? ` ·${g.mentions}` : ''}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {insights.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-1">Insights</p>
                  <ul className="space-y-0.5">
                    {insights.slice(0, 3).map((ins, j) => {
                      const txt = extractInsightText(ins)
                      return txt ? (
                        <li key={j} className="text-xs text-slate-400 line-clamp-2">{txt}</li>
                      ) : null
                    })}
                  </ul>
                </div>
              )}

              {study?.quote && (
                <blockquote className="text-xs text-slate-400 italic border-l-2 border-cyan-500/30 pl-2 line-clamp-3">
                  "{study.quote}"
                  {study.company && <span className="not-italic text-slate-600"> — {study.company}</span>}
                </blockquote>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ---- Displacement Report ----

interface DisplacementMeta {
  total_flows?: number | null
  total_mentions?: number | null
  dominant_driver?: string | null
  pricing_pct?: number | null
  most_displaced_vendor?: string | null
  biggest_winner?: string | null
}

interface DisplacementVendorRow {
  vendor?: string | null
  net_flow?: number | null
  outbound_mentions?: number | null
  inbound_mentions?: number | null
  top_destination?: string | null
  top_source?: string | null
  top_driver?: string | null
}

interface DisplacementBattle {
  from_vendor?: string | null
  to_vendor?: string | null
  mention_count?: number | null
  primary_driver?: string | null
  signal_strength?: string | null
  confidence_score?: number | null
  key_quote?: string | null
  battle_conclusion?: string | null
  durability?: string | null
  source_archetype?: string | null
  target_archetype?: string | null
}

interface DriverSummaryRow {
  driver?: string | null
  mentions?: number | null
  pct?: number | null
  flow_count?: number | null
}

interface DisplacementReportData {
  meta?: DisplacementMeta | null
  market_losers?: DisplacementVendorRow[] | null
  market_winners?: DisplacementVendorRow[] | null
  top_battles?: DisplacementBattle[] | null
  driver_summary?: DriverSummaryRow[] | null
}

function signalColor(strength: string | null | undefined): string {
  const s = (strength ?? '').toLowerCase()
  if (s === 'very_high' || s === 'high') return 'text-red-400'
  if (s === 'medium') return 'text-amber-400'
  return 'text-slate-400'
}

function driverLabel(d: string | null | undefined): string {
  return (d ?? 'unknown').replace(/_/g, ' ')
}

function DriverBadge({ driver }: { driver: string | null | undefined }) {
  if (!driver) return null
  const colors: Record<string, string> = {
    pricing: 'bg-red-900/40 text-red-300',
    price: 'bg-red-900/40 text-red-300',
    features: 'bg-blue-900/40 text-blue-300',
    feature: 'bg-blue-900/40 text-blue-300',
    support: 'bg-yellow-900/40 text-yellow-300',
    integration: 'bg-purple-900/40 text-purple-300',
    usability: 'bg-green-900/40 text-green-300',
    performance: 'bg-cyan-900/40 text-cyan-300',
  }
  const key = driver.toLowerCase().split('_')[0]
  const cls = colors[key] ?? 'bg-slate-800 text-slate-300'
  return (
    <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium whitespace-nowrap', cls)}>
      {driverLabel(driver)}
    </span>
  )
}

function DisplacementReportDetail({ data }: { data: Record<string, unknown> }) {
  const meta = (data.meta ?? {}) as DisplacementMeta
  const losers: DisplacementVendorRow[] = Array.isArray(data.market_losers) ? (data.market_losers as DisplacementVendorRow[]) : []
  const winners: DisplacementVendorRow[] = Array.isArray(data.market_winners) ? (data.market_winners as DisplacementVendorRow[]) : []
  const battles: DisplacementBattle[] = Array.isArray(data.top_battles) ? (data.top_battles as DisplacementBattle[]) : []
  const drivers: DriverSummaryRow[] = Array.isArray(data.driver_summary) ? (data.driver_summary as DriverSummaryRow[]) : []

  const totalMentions = typeof meta.total_mentions === 'number' ? meta.total_mentions : null
  const pricingPct = typeof meta.pricing_pct === 'number' ? `${(meta.pricing_pct * 100).toFixed(0)}%` : null

  return (
    <div className="space-y-5 min-w-0 [overflow-wrap:anywhere]">
      {/* Headline stat strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Flows Tracked', value: meta.total_flows },
          { label: 'Total Mentions', value: totalMentions },
          { label: 'Top Driver', value: driverLabel(meta.dominant_driver) },
          { label: 'Pricing Share', value: pricingPct },
        ].map(({ label, value }) =>
          value != null ? (
            <div key={label} className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3 text-center">
              <p className="text-xs text-slate-400 mb-1">{label}</p>
              <p className="text-lg font-semibold text-white leading-tight">{String(value)}</p>
            </div>
          ) : null,
        )}
      </div>

      {/* Market leaderboard: losers + winners side by side */}
      {(losers.length > 0 || winners.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Losers */}
          {losers.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
              <h3 className="flex items-center gap-1.5 text-sm font-medium text-red-400 mb-3">
                <TrendingDown className="h-4 w-4" /> Market Losers
              </h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700/50">
                    <th className="text-left pb-1.5 font-normal">Vendor</th>
                    <th className="text-right pb-1.5 font-normal">Net</th>
                    <th className="text-left pb-1.5 font-normal pl-3">To</th>
                    <th className="text-left pb-1.5 font-normal pl-2">Driver</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {losers.map((row, i) => (
                    <tr key={i} className="py-1">
                      <td className="py-1.5 text-slate-200 max-w-[6rem] truncate pr-2">{row.vendor ?? '—'}</td>
                      <td className="py-1.5 text-right text-red-400 font-medium whitespace-nowrap">
                        {typeof row.net_flow === 'number' ? `${row.net_flow > 0 ? '+' : ''}${row.net_flow}` : '—'}
                      </td>
                      <td className="py-1.5 pl-3 text-slate-400 max-w-[5rem] truncate">{row.top_destination ?? '—'}</td>
                      <td className="py-1.5 pl-2">
                        <DriverBadge driver={row.top_driver} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Winners */}
          {winners.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
              <h3 className="flex items-center gap-1.5 text-sm font-medium text-green-400 mb-3">
                <Zap className="h-4 w-4" /> Market Winners
              </h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700/50">
                    <th className="text-left pb-1.5 font-normal">Vendor</th>
                    <th className="text-right pb-1.5 font-normal">Net</th>
                    <th className="text-left pb-1.5 font-normal pl-3">From</th>
                    <th className="text-left pb-1.5 font-normal pl-2">Driver</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {winners.map((row, i) => (
                    <tr key={i} className="py-1">
                      <td className="py-1.5 text-slate-200 max-w-[6rem] truncate pr-2">{row.vendor ?? '—'}</td>
                      <td className="py-1.5 text-right text-green-400 font-medium whitespace-nowrap">
                        {typeof row.net_flow === 'number' ? `+${row.net_flow}` : '—'}
                      </td>
                      <td className="py-1.5 pl-3 text-slate-400 max-w-[5rem] truncate">{row.top_source ?? '—'}</td>
                      <td className="py-1.5 pl-2">
                        <DriverBadge driver={row.top_driver} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Driver breakdown */}
      {drivers.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Churn Drivers</h3>
          <div className="space-y-2">
            {drivers.map((d, i) => {
              const pct = typeof d.pct === 'number' ? d.pct : 0
              const barWidth = `${Math.min(pct * 100, 100).toFixed(1)}%`
              return (
                <div key={i} className="flex items-center gap-3 text-xs">
                  <span className="w-32 text-slate-300 shrink-0 capitalize">{driverLabel(d.driver)}</span>
                  <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-2 rounded-full bg-cyan-500/70"
                      style={{ width: barWidth }}
                    />
                  </div>
                  <span className="w-10 text-right text-slate-400 shrink-0">{(pct * 100).toFixed(0)}%</span>
                  {d.mentions != null && (
                    <span className="text-slate-600 shrink-0">{d.mentions}m</span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Top battles */}
      {battles.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="flex items-center gap-1.5 text-sm font-medium text-slate-300 mb-3">
            <Swords className="h-4 w-4 text-amber-400" /> Top Battles
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {battles.map((b, i) => (
              <div key={i} className="bg-slate-800/50 border border-slate-700/40 rounded-lg p-3 space-y-2">
                {/* Header: from → to */}
                <div className="flex items-center gap-1.5 text-sm font-medium flex-wrap">
                  <span className="text-red-300">{b.from_vendor ?? '?'}</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-green-300">{b.to_vendor ?? '?'}</span>
                </div>
                {/* Chips row */}
                <div className="flex flex-wrap gap-1.5 items-center">
                  {b.mention_count != null && (
                    <span className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300 text-xs">{b.mention_count} mentions</span>
                  )}
                  {b.signal_strength && (
                    <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium', signalColor(b.signal_strength))}>
                      {b.signal_strength.replace(/_/g, ' ')}
                    </span>
                  )}
                  <DriverBadge driver={b.primary_driver} />
                </div>
                {/* Quote */}
                {b.key_quote && (
                  <blockquote className="text-xs text-slate-400 italic border-l-2 border-cyan-500/40 pl-2 break-words whitespace-pre-wrap">
                    "{b.key_quote}"
                  </blockquote>
                )}
                {/* Conclusion */}
                {b.battle_conclusion && (
                  <p className="text-xs text-slate-300 leading-relaxed">{b.battle_conclusion}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ---- Vendor Scorecard ----

function parseCompetitor(threat: string | null | undefined): string {
  if (!threat) return '—'
  return threat.replace(/\s*\(\d+\s*mentions?\)/i, '').trim() || '—'
}

function formatSentiment(s: string | null | undefined): string {
  if (!s) return '—'
  const lower = s.toLowerCase()
  if (lower === 'consistently_negative') return 'neg'
  if (lower === 'consistently_positive') return 'pos'
  if (lower === 'mostly_negative') return 'mostly neg'
  if (lower === 'mostly_positive') return 'mostly pos'
  return s.replace(/_/g, ' ')
}

function sentimentColor(s: string | null | undefined): string {
  if (!s) return 'text-slate-400'
  const lower = s.toLowerCase()
  if (lower.includes('negative')) return 'text-red-400'
  if (lower.includes('positive')) return 'text-green-400'
  if (lower.includes('improving')) return 'text-cyan-400'
  return 'text-slate-400'
}

function signalDensityColor(v: number | null | undefined): string {
  if (v == null) return 'text-slate-400'
  if (v >= 20) return 'text-red-400 font-medium'
  if (v >= 12) return 'text-amber-400'
  return 'text-green-400'
}

function urgencyColor(v: number | null | undefined): string {
  if (v == null) return 'text-slate-400'
  if (v >= 3.5) return 'text-red-400'
  if (v >= 2.5) return 'text-amber-400'
  return 'text-slate-300'
}

function VendorScorecardDetail({ items }: { items: VendorScorecardViewModel[] }) {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 overflow-x-auto min-w-0">
      <table className="w-full text-xs min-w-[640px]">
        <thead>
          <tr className="text-slate-500 border-b border-slate-700/50 text-left">
            <th className="pb-2 font-normal pr-4">Vendor</th>
            <th className="pb-2 font-normal pr-3 text-right">Reviews</th>
            <th className="pb-2 font-normal pr-3 text-right">Signal</th>
            <th className="pb-2 font-normal pr-3 text-right">Urgency</th>
            <th className="pb-2 font-normal pr-3">Top Pain</th>
            <th className="pb-2 font-normal pr-3">Competitor</th>
            <th className="pb-2 font-normal pr-3">Trend</th>
            <th className="pb-2 font-normal">Sentiment</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {items.map((item, i) => (
            <tr key={i} className="hover:bg-slate-800/20 transition-colors">
              <td className="py-1.5 pr-4 text-slate-200 font-medium whitespace-nowrap">{item.vendor ?? '—'}</td>
              <td className="py-1.5 pr-3 text-right text-slate-300">{item.total_reviews ?? '—'}</td>
              <td className={clsx('py-1.5 pr-3 text-right', signalDensityColor(item.churn_signal_density))}>
                {item.churn_signal_density ?? '—'}
              </td>
              <td className={clsx('py-1.5 pr-3 text-right', urgencyColor(item.avg_urgency))}>
                {item.avg_urgency ?? '—'}
              </td>
              <td className="py-1.5 pr-3">
                {item.top_pain ? <DriverBadge driver={item.top_pain} /> : <span className="text-slate-500">—</span>}
              </td>
              <td className="py-1.5 pr-3 text-slate-400 whitespace-nowrap">{parseCompetitor(item.top_competitor_threat)}</td>
              <td className="py-1.5 pr-3 text-slate-400 capitalize whitespace-nowrap">{item.trend ?? '—'}</td>
              <td className={clsx('py-1.5 whitespace-nowrap', sentimentColor(item.sentiment_direction))}>
                {formatSentiment(item.sentiment_direction)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function SpecializedReportData({
  reportType,
  data,
}: {
  reportType: string
  data: unknown
}) {
  const normalizedValue = normalizeUnknown(data, reportType)
  const normalized = normalizeReportObject(Array.isArray(normalizedValue) ? {} : (normalizedValue as Record<string, unknown> | null | undefined))
  if (reportType === 'challenger_brief') return <ChallengerBriefDetail data={toChallengerBriefViewModel(normalized)} />
  if (reportType === 'accounts_in_motion') return <AccountsInMotionDetail data={toAccountsInMotionViewModel(normalized)} />
  if (reportType === 'battle_card') return <BattleCardDetail data={toBattleCardViewModel(normalized)} rawData={normalized} />
  if (reportType === 'vendor_comparison' || reportType === 'account_comparison') {
    return <ComparisonReportDetail data={toComparisonReportViewModel(normalized)} rawData={normalized} />
  }
  if (reportType === 'weekly_churn_feed') return <WeeklyChurnFeedDetail items={toWeeklyChurnFeedItems(normalizedValue)} />
  if (reportType === 'vendor_scorecard') return <VendorScorecardDetail items={toVendorScorecards(normalizedValue)} />
  if (reportType === 'displacement_report') return <DisplacementReportDetail data={normalized} />
  if (reportType === 'category_overview') {
    const items: CategoryOverviewItem[] = Array.isArray(normalizedValue)
      ? (normalizedValue as CategoryOverviewItem[])
      : Array.isArray(normalized.category_overview)
        ? (normalized.category_overview as CategoryOverviewItem[])
        : []
    return <CategoryOverviewDetail items={items} />
  }
  return null
}
