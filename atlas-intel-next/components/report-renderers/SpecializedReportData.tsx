import { clsx } from 'clsx'
import { Clock, Crosshair, MessageSquareQuote, Shield, Swords, Target, TrendingDown, Users, Zap } from 'lucide-react'
import type { ReactNode } from 'react'
import ArchetypeBadge from '@/components/ArchetypeBadge'
import { StructuredReportData } from '@/components/report-renderers/StructuredReportData'
import { normalizeReportObject, normalizeUnknown } from '@/lib/reportNormalization'
import { useState } from 'react'
import {
  toBattleCardViewModel,
  toAccountsInMotionViewModel,
  toChallengerBriefViewModel,
  toComparisonReportViewModel,
  toVendorDeepDives,
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
  VendorDeepDiveViewModel,
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

      {data.account_pressure_summary && (
        <SectionCard title="Account Pressure" icon={<Zap className="h-4 w-4 text-orange-400" />}>
          <p className="text-sm text-slate-300">{data.account_pressure_summary}</p>
          {data.account_pressure_metrics && (
            <div className="grid grid-cols-3 gap-2 mt-2">
              <MetricRow label="Total Accounts" value={data.account_pressure_metrics.total_accounts} color="text-orange-400" />
              <MetricRow label="High Intent" value={data.account_pressure_metrics.high_intent_count} color="text-red-400" />
              <MetricRow label="Active Eval" value={data.account_pressure_metrics.active_eval_count} color="text-amber-400" />
            </div>
          )}
          {data.priority_account_names && data.priority_account_names.length > 0 && (
            <p className="text-xs text-slate-400 mt-2">Priority: {data.priority_account_names.join(', ')}</p>
          )}
        </SectionCard>
      )}

      {data.timing_summary && (
        <SectionCard title="Timing Intelligence" icon={<Clock className="h-4 w-4 text-blue-400" />}>
          <p className="text-sm text-slate-300">{data.timing_summary}</p>
          {data.priority_timing_triggers && data.priority_timing_triggers.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {data.priority_timing_triggers.map((trigger, i) => (
                <span key={i} className="px-2 py-0.5 text-xs bg-blue-900/30 text-blue-300 rounded">{trigger}</span>
              ))}
            </div>
          )}
        </SectionCard>
      )}

      {data.segment_targeting_summary && (
        <SectionCard title="Segment Targeting" icon={<Crosshair className="h-4 w-4 text-purple-400" />}>
          <p className="text-sm text-slate-300">{data.segment_targeting_summary}</p>
        </SectionCard>
      )}

      {data.category_council && (data.category_council.conclusion || data.category_council.market_regime) && (
        <SectionCard title="Category Council" icon={<Users className="h-4 w-4 text-emerald-400" />}>
          <div className="space-y-1">
            {data.category_council.market_regime && <MetricRow label="Market Regime" value={data.category_council.market_regime.replace(/_/g, ' ')} />}
            {data.category_council.winner && <MetricRow label="Category Winner" value={data.category_council.winner} color="text-green-400" />}
            {data.category_council.loser && <MetricRow label="Losing Ground" value={data.category_council.loser} color="text-red-400" />}
          </div>
          {data.category_council.conclusion && <p className="text-sm text-slate-300 mt-2">{data.category_council.conclusion}</p>}
          {data.category_council.durability && <p className="text-xs text-slate-400 mt-1">Durability: {data.category_council.durability}</p>}
        </SectionCard>
      )}

      {accounts.length > 0 && (
        <SectionCard title={`Prospecting List (${accounts.length} accounts)`} icon={<Target className="h-4 w-4 text-amber-400" />}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm table-auto">
              <thead>
                <tr className="border-b border-slate-700/50">
                  {['Company', 'Score', 'Stage', 'Urg', 'Pain', 'DM', 'Conf', 'Alts'].map((header) => (
                    <th key={header} className="text-left text-xs text-slate-400 px-2 py-1 align-top break-words">{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {accounts.slice(0, 25).map((account: AccountsInMotionAccountViewModel, index: number) => (
                  <tr key={index} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                    <td className="px-2 py-1 text-slate-300 align-top break-words">
                      {(account.company ?? '').slice(0, 24)}
                      {account.contact_name && <span className="block text-[10px] text-cyan-400">{account.contact_name}</span>}
                      {account.contact_title && <span className="block text-[10px] text-slate-500">{account.contact_title}</span>}
                      {account.quality_flags && account.quality_flags.length > 0 && (
                        <span className="flex flex-wrap gap-0.5 mt-0.5">{account.quality_flags.slice(0, 2).map((flag, fi) => (
                          <span key={fi} className="px-1 py-0 text-[9px] bg-slate-800 text-slate-400 rounded">{flag.replace(/_/g, ' ')}</span>
                        ))}</span>
                      )}
                    </td>
                    <td className="px-2 py-1">
                      <span className={clsx('font-medium', Number(account.opportunity_score) >= 50 ? 'text-green-400' : Number(account.opportunity_score) >= 30 ? 'text-amber-400' : 'text-slate-300')}>
                        {account.opportunity_score ?? ''}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{(account.buying_stage ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-slate-300 align-top break-words">{typeof account.urgency === 'number' ? account.urgency.toFixed(0) : ''}</td>
                    <td className="px-2 py-1 text-slate-400 align-top break-words">{(account.pain_category ?? '').replace(/_/g, ' ')}</td>
                    <td className="px-2 py-1 text-center">{account.decision_maker ? <span className="text-green-400 text-xs">DM</span> : ''}</td>
                    <td className="px-2 py-1 text-xs text-slate-500 align-top">{typeof account.confidence === 'number' ? `${Math.round(account.confidence * 100)}%` : ''}</td>
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
                      &ldquo;{account.top_quote}&rdquo;
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
  const hiCompanies = Array.isArray(rawData.high_intent_companies)
    ? (rawData.high_intent_companies as Record<string, unknown>[])
    : []

  const skipKeys = [
    // Explicitly rendered fields
    'vendor', 'category', 'churn_pressure_score', 'total_reviews', 'confidence',
    'archetype', 'archetype_risk_level', 'archetype_key_signals', 'archetype_confidence',
    'vendor_weaknesses', 'weakness_analysis', 'customer_pain_quotes',
    'competitor_differentiators', 'cross_vendor_battles', 'competitive_landscape',
    'resource_asymmetry', 'category_council', 'objection_handlers', 'talk_track',
    'recommended_plays', 'active_evaluation_deadlines', 'source_distribution',
    'llm_render_status', 'quality_status', 'quality_score', 'battle_card_quality',
    'data_stale', 'objection_data', 'high_intent_companies',
    'executive_summary', 'discovery_questions', 'landmine_questions',
    'account_pressure_metrics', 'buyer_authority', 'integration_stack',
    'keyword_spikes', 'retention_signals', 'incumbent_strengths',
    'falsification_conditions', 'uncertainty_sources', 'evidence_conclusions',
    'low_confidence_sections', 'evidence_depth_warning',
    // Internal LLM reasoning contracts — too nested to show usefully
    'vendor_core_reasoning', 'displacement_reasoning', 'category_reasoning',
    'account_reasoning', 'reasoning_contracts', 'locked_facts',
    'render_packet_version', 'render_contracts_used', 'render_packet_hash',
    'segment_playbook', 'timing_intelligence', 'timing_summary',
    'account_pressure_summary', 'segment_targets', 'timing_window', 'timing_triggers',
    // Internal / metadata — not shown in UI
    'synthesis_wedge', 'synthesis_wedge_label', 'risk_level',
    'ecosystem_context',
    'evidence_window', 'evidence_window_days',
    'reasoning_source', 'synthesis_schema_version', 'section_disclaimers',
    'category_dynamics', 'timing_metrics',
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

      {/* Executive Summary — prominent text block */}
      {typeof rawData.executive_summary === 'string' && rawData.executive_summary && (
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-xl p-4">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Executive Summary</p>
          <p className="text-sm text-slate-200 leading-relaxed">{rawData.executive_summary}</p>
        </div>
      )}

      {/* Weakness Analysis — full width, items in 2-col grid (text-heavy, needs room) */}
      {weaknesses.length > 0 && (
        <SectionCard title="Weakness Analysis" icon={<TrendingDown className="h-4 w-4 text-red-400" />}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {weaknesses.slice(0, 6).map((weakness: WeaknessAnalysisItemViewModel, index: number) => (
              <div key={index} className="bg-slate-800/50 border border-slate-700/40 rounded-lg p-3 space-y-1.5">
                <p className="text-sm font-medium text-white">{weakness.weakness ?? weakness.area ?? weakness.name ?? ''}</p>
                {weakness.evidence && <p className="text-xs text-slate-400">{weakness.evidence}</p>}
                {weakness.customer_quote && <blockquote className="text-xs text-slate-300 italic border-l-2 border-cyan-500/50 pl-2 break-words">"{weakness.customer_quote}"</blockquote>}
                {weakness.winning_position && <p className="text-xs text-cyan-300">{weakness.winning_position}</p>}
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Competitive Landscape + Competitor Differentiators — side by side only when both present */}
      {(data.competitive_landscape || data.competitor_differentiators.length > 0) && (
        <div className={clsx('grid gap-6', data.competitive_landscape && data.competitor_differentiators.length > 0 ? 'lg:grid-cols-2' : 'grid-cols-1')}>
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

          {data.competitor_differentiators.length > 0 && (
            <SectionCard title="Competitor Differentiators" icon={<Zap className="h-4 w-4 text-green-400" />}>
              <div className="overflow-x-auto">
                <table className="w-full text-sm table-auto">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left text-xs text-slate-400 px-2 py-1">Competitor</th>
                      <th className="text-right text-xs text-slate-400 px-2 py-1">Mentions</th>
                      <th className="text-left text-xs text-slate-400 px-2 py-1">Driver</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.competitor_differentiators.slice(0, 8).map((item: CompetitorDifferentiatorViewModel, index: number) => (
                      <tr key={index} className="border-b border-slate-800/50">
                        <td className="px-2 py-1.5 text-slate-300">{item.competitor ?? ''}</td>
                        <td className="px-2 py-1.5 text-right text-slate-400">{item.mentions ?? item.count ?? ''}</td>
                        <td className="px-2 py-1.5 text-slate-400 break-words">{item.primary_driver ?? item.solves_weakness ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          )}
        </div>
      )}

      {/* Customer Pain Quotes — full width, items in 2-col grid (long quotes need room) */}
      {data.customer_pain_quotes.length > 0 && (
        <SectionCard title="Customer Pain Quotes" icon={<MessageSquareQuote className="h-4 w-4 text-amber-400" />}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {data.customer_pain_quotes.slice(0, 6).map((quote, index) => (
              <div key={index} className="bg-slate-800/50 border border-slate-700/40 rounded-lg p-3 space-y-1.5">
                <blockquote className="text-sm text-slate-300 italic border-l-2 border-amber-500/50 pl-2 break-words">"{quote.quote}"</blockquote>
                <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                  {quote.company && <span>{quote.company}</span>}
                  {quote.role && <span>{quote.role}</span>}
                  {quote.pain_category && <span className="text-slate-400">{quote.pain_category}</span>}
                  {quote.urgency != null && <span className="text-amber-400">urgency {quote.urgency}/10</span>}
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Segment Playbook + Timing Intelligence */}
      {((data.segment_targets?.length ?? 0) > 0 || data.timing_window || (data.timing_triggers?.length ?? 0) > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {(data.segment_targets?.length ?? 0) > 0 && (
            <SectionCard title="Segment Playbook" icon={<Target className="h-4 w-4 text-cyan-400" />}>
              <div className="space-y-3">
                {data.segment_targets.slice(0, 5).map((seg, index) => (
                  <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                    <p className="text-sm font-medium text-white">{seg.segment}</p>
                    {seg.why_vulnerable && <p className="text-xs text-slate-400 mt-1">{seg.why_vulnerable}</p>}
                    {seg.best_opening_angle && <p className="text-xs text-cyan-300 mt-1">{seg.best_opening_angle}</p>}
                    {seg.disqualifier && <p className="text-xs text-rose-400 mt-1">Skip if: {seg.disqualifier}</p>}
                  </div>
                ))}
              </div>
            </SectionCard>
          )}
          {(data.timing_window || (data.timing_triggers?.length ?? 0) > 0) && (
            <SectionCard title="Timing Intelligence" icon={<Zap className="h-4 w-4 text-amber-400" />}>
              {data.timing_window && (
                <div className="mb-3">
                  <p className="text-xs text-slate-400 mb-1">Best Window</p>
                  <p className="text-sm text-slate-300">{data.timing_window}</p>
                </div>
              )}
              {data.timing_summary && <p className="text-xs text-slate-400 mb-3">{data.timing_summary}</p>}
              {(data.timing_triggers?.length ?? 0) > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-slate-400">Immediate Triggers</p>
                  {data.timing_triggers.slice(0, 5).map((t, index) => (
                    <div key={index} className="bg-slate-800/50 rounded p-2">
                      <p className="text-xs text-white">{t.trigger}</p>
                      {t.action && <p className="text-xs text-cyan-300 mt-0.5">{t.action}</p>}
                      {t.urgency && <p className="text-xs text-amber-400 mt-0.5">{t.urgency}</p>}
                    </div>
                  ))}
                </div>
              )}
            </SectionCard>
          )}
        </div>
      )}

      {/* Cross-Vendor Battles + Market Context — side by side only when both present */}
      {(data.cross_vendor_battles.length > 0 || data.resource_asymmetry || data.category_council) && (
        <div className={clsx('grid gap-6', data.cross_vendor_battles.length > 0 && (data.resource_asymmetry || data.category_council) ? 'lg:grid-cols-2' : 'grid-cols-1')}>
          {data.cross_vendor_battles.length > 0 && (
            <SectionCard title="Cross-Vendor Battles" icon={<Swords className="h-4 w-4 text-cyan-400" />}>
              <div className="space-y-2">
                {data.cross_vendor_battles.slice(0, 4).map((battle, index) => (
                  <div key={index} className="bg-slate-800/50 rounded-lg p-3">
                    <div className="flex flex-wrap gap-2 text-xs mb-1.5">
                      {battle.opponent && <span className="text-white font-medium">vs {battle.opponent}</span>}
                      {battle.winner && <span className="px-2 py-0.5 bg-emerald-500/15 text-emerald-300 rounded">winner: {battle.winner}</span>}
                      {battle.loser && <span className="px-2 py-0.5 bg-red-500/15 text-red-300 rounded">loser: {battle.loser}</span>}
                      {battle.confidence != null && battle.confidence > 0 && <span className="px-2 py-0.5 bg-slate-700/50 text-slate-300 rounded">{Math.round(battle.confidence * 100)}% conf.</span>}
                      {battle.durability && <span className="px-2 py-0.5 bg-amber-500/15 text-amber-300 rounded">{battle.durability}</span>}
                    </div>
                    {battle.conclusion && <p className="text-xs text-slate-300">{battle.conclusion}</p>}
                    {battle.key_insights.length > 0 && (
                      <ul className="mt-1 space-y-0.5">
                        {battle.key_insights.slice(0, 2).map((insight, i) => (
                          <li key={i} className="text-xs text-slate-400 flex gap-1.5">
                            <span className="text-cyan-400 shrink-0">-</span>
                            <span>{insight.insight}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </SectionCard>
          )}

          {(data.resource_asymmetry || data.category_council) && (
            <SectionCard title="Market Context" icon={<Shield className="h-4 w-4 text-cyan-400" />}>
              {data.resource_asymmetry?.conclusion && <p className="text-sm text-slate-300 mb-3">{data.resource_asymmetry.conclusion}</p>}
              {data.resource_asymmetry?.resource_advantage && <MetricRow label="Resource Advantage" value={data.resource_asymmetry.resource_advantage} />}
              {data.category_council && (
                <div className="flex flex-wrap gap-2 text-xs mb-2 mt-3">
                  {data.category_council.market_regime && <span className="px-2 py-0.5 bg-indigo-500/15 text-indigo-300 rounded">{data.category_council.market_regime}</span>}
                  {data.category_council.winner && <span className="px-2 py-0.5 bg-emerald-500/15 text-emerald-300 rounded">winner: {data.category_council.winner}</span>}
                  {data.category_council.loser && <span className="px-2 py-0.5 bg-red-500/15 text-red-300 rounded">loser: {data.category_council.loser}</span>}
                  {data.category_council.confidence != null && data.category_council.confidence > 0 && <span className="px-2 py-0.5 bg-slate-700/50 text-slate-300 rounded">{Math.round(data.category_council.confidence * 100)}% conf.</span>}
                  {data.category_council.durability && <span className="px-2 py-0.5 bg-amber-500/15 text-amber-300 rounded">{data.category_council.durability}</span>}
                </div>
              )}
              {data.category_council?.conclusion && <p className="text-sm text-slate-300">{data.category_council.conclusion}</p>}
              {data.category_council?.key_insights.length ? (
                <ul className="space-y-1 mt-3">
                  {data.category_council.key_insights.slice(0, 4).map((insight, index) => (
                    <li key={index} className="text-xs text-slate-400 flex gap-2 min-w-0">
                      <span className="text-cyan-400 shrink-0">-</span>
                      <span className="min-w-0 break-words">{insight.insight}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </SectionCard>
          )}
        </div>
      )}

      {data.active_evaluation_deadlines.length > 0 && (
        <SectionCard title="Target Accounts" icon={<Target className="h-4 w-4 text-red-400" />}>
          {data.account_pressure_summary && <p className="text-sm text-slate-400 mb-3">{data.account_pressure_summary}</p>}
          {data.account_market_summary && (
            <p className="text-sm text-slate-400 mb-3">{data.account_market_summary}</p>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-sm table-auto">
              <thead>
                <tr className="border-b border-slate-700/50">
                  <th className="text-left text-xs text-slate-400 px-2 py-1">Company</th>
                  <th className="text-left text-xs text-slate-400 px-2 py-1">Why Now</th>
                  <th className="text-left text-xs text-slate-400 px-2 py-1">Stage</th>
                  <th className="text-right text-xs text-slate-400 px-2 py-1">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {[...data.active_evaluation_deadlines].sort((a, b) => {
                  const scoreA = (a.urgency ?? 0) + (a.buying_stage === 'evaluation' || a.buying_stage === 'active_purchase' ? 3 : 0) + (a.evaluation_deadline ? 2 : 0) + (a.contract_end ? 1 : 0)
                  const scoreB = (b.urgency ?? 0) + (b.buying_stage === 'evaluation' || b.buying_stage === 'active_purchase' ? 3 : 0) + (b.evaluation_deadline ? 2 : 0) + (b.contract_end ? 1 : 0)
                  return scoreB - scoreA
                }).slice(0, 10).map((item, index) => {
                  const whyNow = item.evaluation_deadline ? 'eval deadline' : item.contract_end ? 'renewal window' : item.buying_stage === 'evaluation' || item.buying_stage === 'active_purchase' ? 'active evaluation' : (item.urgency ?? 0) >= 7 ? 'high pain + urgency' : item.pain ? item.pain.replace(/_/g, ' ') : ''
                  return (
                    <tr key={index} className="border-b border-slate-800/50">
                      <td className="px-2 py-1 text-slate-300 align-top">{item.company ?? ''}</td>
                      <td className="px-2 py-1 text-cyan-400 align-top text-xs">{whyNow}</td>
                      <td className="px-2 py-1 text-slate-400 align-top">{(item.buying_stage ?? item.decision_timeline ?? '').replace(/_/g, ' ')}</td>
                      <td className="px-2 py-1 text-right text-amber-400 align-top">{item.urgency ?? ''}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Evidence Posture */}
      {(data.evidence_depth_warning || data.low_confidence_sections.length > 0 || data.uncertainty_sources.length > 0 || data.falsification_conditions.length > 0 || data.evidence_conclusions.length > 0) && (
        <SectionCard title="Evidence Posture" icon={<Shield className="h-4 w-4 text-amber-400" />}>
          {data.evidence_depth_warning && (
            <p className="text-xs text-amber-300 mb-3">{data.evidence_depth_warning}</p>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            {data.evidence_conclusions.length > 0 && (
              <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-3">
                <p className="text-emerald-400 font-medium mb-1.5">Safe to Lead With</p>
                <ul className="space-y-1">{data.evidence_conclusions.map((s, i) => (
                  <li key={i} className="text-slate-300">{s}</li>
                ))}</ul>
              </div>
            )}
            {(data.low_confidence_sections.length > 0 || data.uncertainty_sources.length > 0) && (
              <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-3">
                <p className="text-amber-400 font-medium mb-1.5">Use Carefully</p>
                <ul className="space-y-1">
                  {data.low_confidence_sections.map((s, i) => (
                    <li key={`lc-${i}`} className="text-slate-400 flex gap-1.5"><span className="text-amber-400 shrink-0">!</span><span>{s}</span></li>
                  ))}
                  {data.uncertainty_sources.map((s, i) => (
                    <li key={`us-${i}`} className="text-slate-400 flex gap-1.5"><span className="text-slate-500 shrink-0">?</span><span>{s}</span></li>
                  ))}
                </ul>
              </div>
            )}
            {data.falsification_conditions.length > 0 && (
              <div className="bg-rose-500/5 border border-rose-500/20 rounded-lg p-3">
                <p className="text-rose-400 font-medium mb-1.5">Could Break This Thesis</p>
                <ul className="space-y-1">{data.falsification_conditions.map((s, i) => (
                  <li key={i} className="text-slate-400 flex gap-1.5"><span className="text-rose-400 shrink-0">x</span><span>{s}</span></li>
                ))}</ul>
              </div>
            )}
          </div>
        </SectionCard>
      )}

      {/* Operational Signals */}
      {(data.account_pressure_metrics || data.keyword_spikes || data.integration_stack.length > 0 || data.buyer_authority) && (
        <SectionCard title="Operational Signals" icon={<Zap className="h-4 w-4 text-cyan-400" />}>
          <div className="flex flex-wrap gap-3 text-xs">
            {data.account_pressure_metrics?.active_eval_count != null && (
              <div className="bg-slate-800/50 rounded-lg px-3 py-2 text-center">
                <p className="text-lg font-bold text-amber-400">{data.account_pressure_metrics.active_eval_count}</p>
                <p className="text-slate-500">Active Eval</p>
              </div>
            )}
            {data.account_pressure_metrics?.high_intent_count != null && (
              <div className="bg-slate-800/50 rounded-lg px-3 py-2 text-center">
                <p className="text-lg font-bold text-red-400">{data.account_pressure_metrics.high_intent_count}</p>
                <p className="text-slate-500">High Intent</p>
              </div>
            )}
            {data.keyword_spikes?.spike_count != null && data.keyword_spikes.spike_count > 0 && (
              <div className="bg-slate-800/50 rounded-lg px-3 py-2 text-center">
                <p className="text-lg font-bold text-cyan-400">{data.keyword_spikes.spike_count}</p>
                <p className="text-slate-500">Keyword Spikes</p>
              </div>
            )}
          </div>
          {data.buyer_authority && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {(() => {
                const ba = data.buyer_authority
                const pills: Array<[string, string]> = []
                // Handle nested maps: role_types, buying_stages
                const roleTypes = ba.role_types as Record<string, number> | undefined
                const buyingStages = ba.buying_stages as Record<string, number> | undefined
                if (roleTypes && typeof roleTypes === 'object') {
                  Object.entries(roleTypes)
                    .filter(([k, v]) => k !== 'unknown' && typeof v === 'number' && v > 0)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .slice(0, 3)
                    .forEach(([k, v]) => pills.push([k.replace(/_/g, ' '), String(v)]))
                }
                if (buyingStages && typeof buyingStages === 'object') {
                  Object.entries(buyingStages)
                    .filter(([, v]) => typeof v === 'number' && v > 0)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .slice(0, 3)
                    .forEach(([k, v]) => pills.push([k.replace(/_/g, ' '), String(v)]))
                }
                // Flat keys fallback
                if (pills.length === 0) {
                  if (ba.dominant_role) pills.push(['role', String(ba.dominant_role)])
                  if (ba.buying_stage) pills.push(['stage', String(ba.buying_stage)])
                  if (typeof ba.dm_rate === 'number') pills.push(['DM rate', `${Math.round(Number(ba.dm_rate) * 100)}%`])
                }
                return pills.map(([label, val], i) => (
                  <span key={i} className="px-1.5 py-0.5 bg-indigo-500/10 text-indigo-300 rounded text-xs">{label}: {val}</span>
                ))
              })()}
            </div>
          )}
          {data.keyword_spikes?.keywords && data.keyword_spikes.keywords.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {data.keyword_spikes.keywords.slice(0, 8).map((kw, i) => (
                <span key={i} className="px-1.5 py-0.5 bg-cyan-500/10 text-cyan-300 rounded text-xs">{kw}</span>
              ))}
            </div>
          )}
          {data.integration_stack.length > 0 && (
            <div className="mt-2">
              <p className="text-xs text-slate-500 mb-1">Integration Stack</p>
              <div className="flex flex-wrap gap-1.5">
                {data.integration_stack.slice(0, 8).map((int, i) => (
                  <span key={i} className="px-1.5 py-0.5 bg-slate-700/50 text-slate-300 rounded text-xs">{int}</span>
                ))}
              </div>
            </div>
          )}
        </SectionCard>
      )}

      {/* Retention Signals + Incumbent Strengths */}
      {(data.retention_signals.length > 0 || data.incumbent_strengths.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {data.retention_signals.length > 0 && (
            <SectionCard title="Retention Signals" icon={<Shield className="h-4 w-4 text-emerald-400" />}>
              <div className="space-y-1">
                {data.retention_signals.map((sig, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-slate-300">{sig.aspect}</span>
                    <span className="text-slate-500">{sig.mentions ?? ''}</span>
                  </div>
                ))}
              </div>
            </SectionCard>
          )}
          {data.incumbent_strengths.length > 0 && (
            <SectionCard title="Incumbent Strengths" icon={<Shield className="h-4 w-4 text-cyan-400" />}>
              <div className="space-y-1">
                {data.incumbent_strengths.map((str, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-slate-300">{str.area}</span>
                    <div className="flex items-center gap-2">
                      {str.source && <span className="text-slate-500">{str.source}</span>}
                      <span className="text-slate-500">{str.mention_count ?? ''}</span>
                    </div>
                  </div>
                ))}
              </div>
            </SectionCard>
          )}
        </div>
      )}

      {/* Landmine + Discovery Questions */}
      {(data.landmine_questions.length > 0 || data.discovery_questions.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {data.landmine_questions.length > 0 && (
            <SectionCard title="Landmine Questions" icon={<Shield className="h-4 w-4 text-rose-400" />}>
              <ul className="space-y-2">
                {data.landmine_questions.map((q, i) => (
                  <li key={i} className="text-sm text-slate-300 flex gap-2 min-w-0">
                    <span className="text-rose-400 shrink-0">{i + 1}.</span>
                    <span className="min-w-0 break-words">{q}</span>
                  </li>
                ))}
              </ul>
            </SectionCard>
          )}
          {data.discovery_questions.length > 0 && (
            <SectionCard title="Discovery Questions" icon={<MessageSquareQuote className="h-4 w-4 text-cyan-400" />}>
              <ul className="space-y-2">
                {data.discovery_questions.map((q, i) => (
                  <li key={i} className="text-sm text-slate-300 flex gap-2 min-w-0">
                    <span className="text-cyan-400 shrink-0">{i + 1}.</span>
                    <span className="min-w-0 break-words">{q}</span>
                  </li>
                ))}
              </ul>
            </SectionCard>
          )}
        </div>
      )}

      {/* Recommended Plays — full width, items in 2-col grid (text-heavy cards need room) */}
      {data.recommended_plays.length > 0 && (
        <SectionCard title="Recommended Plays" icon={<Target className="h-4 w-4 text-cyan-400" />}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {data.recommended_plays.slice(0, 6).map((play: RecommendedPlayViewModel, index: number) => (
              <div key={index} className="bg-slate-800/50 border border-slate-700/40 rounded-lg p-3 space-y-1.5">
                <p className="text-sm font-medium text-white">{play.play ?? play.name ?? play.description ?? ''}</p>
                {play.target_segment && <p className="text-xs text-amber-300">{play.target_segment}</p>}
                {play.key_message && <p className="text-xs text-slate-400">{play.key_message}</p>}
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Talk Track — full width, stages in a 3-col row */}
      {data.talk_track && (
        <SectionCard title="Talk Track" icon={<MessageSquareQuote className="h-4 w-4 text-cyan-400" />}>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {([
              { key: 'opening' as const, label: 'Opening' },
              { key: 'mid_call_pivot' as const, label: 'Mid-Call Pivot' },
              { key: 'closing' as const, label: 'Closing' },
            ] as const).map(({ key, label }) => {
              const value = data.talk_track?.[key]
              if (!value) return null
              return (
                <div key={key} className="bg-slate-800/50 border border-slate-700/40 rounded-lg p-3">
                  <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">{label}</p>
                  <p className="text-sm text-slate-300 break-words">{value}</p>
                </div>
              )
            })}
          </div>
        </SectionCard>
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

      {(() => {
        const obj = rawData.objection_data !== null && typeof rawData.objection_data === 'object' && !Array.isArray(rawData.objection_data)
          ? (rawData.objection_data as Record<string, unknown>)
          : null
        if (!obj) return null
        const stats: { label: string; value: string; accent?: string }[] = []
        if (obj.avg_urgency != null) stats.push({ label: 'Avg Urgency', value: String(obj.avg_urgency), accent: 'text-amber-400' })
        if (obj.recommend_ratio != null) stats.push({ label: 'Recommend Ratio', value: typeof obj.recommend_ratio === 'number' ? `${obj.recommend_ratio}%` : String(obj.recommend_ratio) })
        if (obj.positive_review_pct != null) stats.push({ label: 'Positive Review %', value: typeof obj.positive_review_pct === 'number' ? `${(obj.positive_review_pct as number).toFixed(1)}%` : String(obj.positive_review_pct) })
        if (obj.dm_churn_rate != null) stats.push({ label: 'DM Churn Rate', value: String(obj.dm_churn_rate) })
        if (obj.sentiment_direction != null) stats.push({ label: 'Sentiment', value: String(obj.sentiment_direction).replace(/_/g, ' ') })
        if (!stats.length) return null
        return (
          <SectionCard title="Objection Metrics" icon={<Shield className="h-4 w-4 text-slate-400" />}>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              {stats.map(({ label, value, accent }) => (
                <div key={label} className="bg-slate-800/60 border border-slate-700/40 rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-400 mb-1">{label}</p>
                  <p className={`text-sm font-semibold leading-tight ${accent ?? 'text-white'}`}>{value}</p>
                </div>
              ))}
            </div>
          </SectionCard>
        )
      })()}

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

            {/* Named Accounts + Retention Strengths side by side */}
            {(item.named_accounts.length > 0 || item.retention_strengths.length > 0) && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {item.named_accounts.length > 0 && (
                  <div>
                    <p className="text-xs text-slate-500 mb-1.5">Target Accounts</p>
                    <div className="space-y-1">
                      {item.named_accounts.slice(0, 5).map((acct, j) => (
                        <div key={j} className="text-xs space-y-0.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-slate-300 truncate">{acct.company ?? ''}</span>
                            <div className="flex items-center gap-1.5 shrink-0">
                              {acct.decision_maker && <span className="text-cyan-400">DM</span>}
                              {acct.confidence_score != null && acct.confidence_score > 0 && <span className="text-slate-500">{Math.round(acct.confidence_score * 100)}%</span>}
                              {acct.urgency != null && <span className="text-amber-400">{acct.urgency}</span>}
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-x-2 text-slate-500">
                            {acct.title && <span>{acct.title}</span>}
                            {acct.buying_stage && <span>{acct.buying_stage.replace(/_/g, ' ')}</span>}
                            {acct.source && <span>{acct.source}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {item.retention_strengths.length > 0 && (
                  <div>
                    <p className="text-xs text-slate-500 mb-1.5">Retention Strengths</p>
                    <div className="space-y-1">
                      {item.retention_strengths.slice(0, 3).map((s, j) => (
                        <div key={j} className="flex items-center justify-between text-xs">
                          <span className="text-emerald-300 capitalize">{(s.area ?? '').replace(/_/g, ' ')}</span>
                          <span className="text-slate-500">{s.mention_count ?? ''}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Category Council */}
            {item.category_council?.conclusion && (
              <div className="text-xs">
                <div className="flex flex-wrap gap-1.5 mb-1">
                  {item.category_council.market_regime && <span className="px-1.5 py-0.5 bg-indigo-500/15 text-indigo-300 rounded">{item.category_council.market_regime}</span>}
                  {item.category_council.winner && <span className="px-1.5 py-0.5 bg-emerald-500/15 text-emerald-300 rounded">winner: {item.category_council.winner}</span>}
                  {item.category_council.loser && <span className="px-1.5 py-0.5 bg-red-500/15 text-red-300 rounded">loser: {item.category_council.loser}</span>}
                  {item.category_council.confidence != null && item.category_council.confidence > 0 && <span className="px-1.5 py-0.5 bg-slate-700/50 text-slate-300 rounded">{Math.round(item.category_council.confidence * 100)}%</span>}
                  {item.category_council.durability && <span className="px-1.5 py-0.5 bg-amber-500/15 text-amber-300 rounded">{item.category_council.durability}</span>}
                </div>
                <p className="text-slate-400">{item.category_council.conclusion}</p>
                {item.category_council.key_insights.length > 0 && (
                  <ul className="mt-1 space-y-0.5">
                    {item.category_council.key_insights.slice(0, 2).map((insight, k) => (
                      <li key={k} className="text-slate-500 flex gap-1.5"><span className="text-cyan-400 shrink-0">-</span><span>{insight.insight}</span></li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* Timing + Account pressure summaries */}
            {(item.timing_summary || item.account_pressure_summary || item.priority_timing_triggers.length > 0) && (
              <div className="space-y-1 text-xs text-slate-400">
                {item.timing_summary && <p>{item.timing_summary}</p>}
                {item.priority_timing_triggers.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {item.priority_timing_triggers.slice(0, 3).map((t, k) => (
                      <span key={k} className="px-1.5 py-0.5 bg-amber-500/10 text-amber-300 rounded text-xs">{t}</span>
                    ))}
                  </div>
                )}
                {item.account_pressure_summary && <p>{item.account_pressure_summary}</p>}
              </div>
            )}

            {/* Secondary metrics row */}
            {(item.sentiment_direction || item.trend || item.dm_churn_rate != null || item.price_complaint_rate != null) && (
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500">
                {item.sentiment_direction && <span>Sentiment: {item.sentiment_direction}</span>}
                {item.trend && <span>Trend: {item.trend}</span>}
                {item.dm_churn_rate != null && <span>DM churn: {(item.dm_churn_rate * 100).toFixed(0)}%</span>}
                {item.price_complaint_rate != null && <span>Price complaints: {(item.price_complaint_rate * 100).toFixed(0)}%</span>}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ---- Vendor Deep Dive ----

function VendorDeepDiveDetail({ items }: { items: VendorDeepDiveViewModel[] }) {
  const [selected, setSelected] = useState(0)
  if (items.length === 0) return <p className="text-slate-500 text-sm">No vendor data found.</p>

  const v = items[selected]
  const pains = v.pain_breakdown ?? []
  const maxPain = Number(pains[0]?.count ?? 1)
  const targets = v.displacement_targets ?? []
  const maxTarget = Number(targets[0]?.mention_count ?? 1)
  const gaps = v.feature_gaps ?? []
  const industries = v.industry_distribution ?? []
  const sizes = v.company_size_distribution ?? []
  const studies = v.case_studies ?? []
  const sent = v.sentiment_breakdown ?? {}
  const sentTotal = (sent.positive ?? 0) + (sent.negative ?? 0) + (sent.neutral ?? 0)
  const retentionStrengths = v.retention_strengths ?? []
  const council = v.category_council

  return (
    <div className="space-y-4 min-w-0">
      {/* Vendor picker */}
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={selected}
          onChange={e => setSelected(Number(e.target.value))}
          className="bg-slate-800 border border-slate-600 text-slate-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-500"
        >
          {items.map((item, i) => (
            <option key={i} value={i}>
              {item.vendor ?? `Vendor ${i + 1}`}
              {item.risk_level ? ` -- ${item.risk_level}` : ''}
            </option>
          ))}
        </select>
        <span className="text-xs text-slate-500">{items.length} vendors · sorted by churn pressure</span>
      </div>

      {/* Vendor header */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-start justify-between gap-3 flex-wrap mb-3">
          <div>
            <h2 className="text-lg font-bold text-slate-100">{v.vendor ?? '--'}</h2>
            <p className="text-xs text-slate-400 mt-0.5">{v.category ?? '--'}</p>
          </div>
          <div className="flex flex-wrap gap-2 items-center">
            {v.risk_level && (
              <span className={clsx(
                'px-2 py-0.5 rounded text-xs font-semibold',
                v.risk_level === 'high' ? 'bg-red-900/50 text-red-300' :
                v.risk_level === 'medium' ? 'bg-amber-900/50 text-amber-300' :
                'bg-green-900/50 text-green-300'
              )}>
                {v.risk_level} risk
              </span>
            )}
            {v.archetype && (
              <span className="px-2 py-0.5 rounded bg-purple-900/40 text-purple-300 text-xs">
                {v.archetype.replace(/_/g, ' ')}
                {v.archetype_confidence != null ? ` · ${(v.archetype_confidence * 100).toFixed(0)}%` : ''}
              </span>
            )}
          </div>
        </div>

        {/* Stat strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Pressure', value: v.churn_pressure_score != null ? `${Math.round(v.churn_pressure_score)}/100` : null, color: 'text-red-400' },
            { label: 'Signal Density', value: v.churn_signal_density != null ? `${v.churn_signal_density}%` : null, color: 'text-amber-400' },
            { label: 'Avg Urgency', value: v.avg_urgency, color: 'text-orange-400' },
            { label: 'Reviews', value: v.total_reviews, color: 'text-slate-300' },
          ].map(({ label, value, color }) => value != null ? (
            <div key={label} className="bg-slate-800/60 rounded-lg p-2.5 text-center">
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <p className={clsx('text-base font-semibold', color)}>{String(value)}</p>
            </div>
          ) : null)}
        </div>

        {/* Secondary metrics row */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-xs text-slate-400">
          {v.dm_churn_rate != null && <span>DM churn: <span className="text-slate-200">{(v.dm_churn_rate * 100).toFixed(0)}%</span></span>}
          {v.price_complaint_rate != null && <span>Price complaints: <span className="text-slate-200">{(v.price_complaint_rate * 100).toFixed(0)}%</span></span>}
          {v.sentiment_direction && <span>Sentiment: <span className="text-slate-200">{v.sentiment_direction.replace(/_/g, ' ')}</span></span>}
          {v.trend && <span>Trend: <span className="text-slate-200">{v.trend}</span></span>}
          {v.dominant_buyer_role && v.dominant_buyer_role !== 'unknown' && <span>Buyer role: <span className="text-slate-200">{v.dominant_buyer_role.replace(/_/g, ' ')}</span></span>}
        </div>
      </div>

      {/* Pain + Displacement */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {pains.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Pain Breakdown</h3>
            <div className="space-y-2">
              {pains.map((p, i) => {
                const label = (p.category ?? '').replace(/_/g, ' ')
                const count = Number(p.count ?? 0)
                const pct = maxPain > 0 ? Math.min((count / maxPain) * 100, 100) : 0
                return (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className="w-32 text-slate-300 shrink-0 capitalize truncate">{label}</span>
                    <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div className="h-2 rounded-full bg-red-500/60" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="w-16 text-right text-slate-400 shrink-0">
                      {count}{p.pct != null ? ` · ${(p.pct * 100).toFixed(0)}%` : ''}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {targets.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Displacement Targets</h3>
            <div className="space-y-2">
              {targets.map((t, i) => {
                const count = Number(t.mention_count ?? 0)
                const pct = maxTarget > 0 ? Math.min((count / maxTarget) * 100, 100) : 0
                return (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className="w-28 text-green-400 shrink-0 truncate">{t.vendor ?? '--'}</span>
                    <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div className="h-2 rounded-full bg-green-500/50" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="text-slate-500 shrink-0 w-8 text-right">{count}</span>
                    {t.primary_driver && <DriverBadge driver={t.primary_driver} />}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Feature gaps + Customer profile */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {gaps.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Feature Gaps</h3>
            <div className="flex flex-wrap gap-1.5">
              {gaps.map((g, i) => (
                <span key={i} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-xs">
                  {g.feature ?? '?'}{g.mentions != null ? <span className="text-slate-500"> ·{g.mentions}</span> : null}
                </span>
              ))}
            </div>
          </div>
        )}

        {(industries.length > 0 || sizes.length > 0) && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Customer Profile</h3>
            {industries.length > 0 && (
              <div className="mb-2">
                <p className="text-xs text-slate-500 mb-1">Industries</p>
                <div className="flex flex-wrap gap-1">
                  {industries.slice(0, 5).map((ind, i) => (
                    <span key={i} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-xs capitalize">
                      {ind.industry ?? '?'}{ind.count != null ? ` ·${ind.count}` : ''}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {sizes.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 mb-1">Company Size</p>
                <div className="flex flex-wrap gap-1">
                  {sizes.map((sz, i) => (
                    <span key={i} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-xs">
                      {sz.size ?? '?'}{sz.count != null ? ` ·${sz.count}` : ''}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Sentiment breakdown */}
      {sentTotal > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Sentiment Distribution</h3>
          <div className="flex gap-2 text-xs items-center">
            {[
              { label: 'Positive', count: sent.positive ?? 0, cls: 'bg-green-500/60' },
              { label: 'Neutral', count: sent.neutral ?? 0, cls: 'bg-slate-600' },
              { label: 'Negative', count: sent.negative ?? 0, cls: 'bg-red-500/60' },
            ].map(({ label, count, cls }) => {
              const pct = sentTotal > 0 ? (count / sentTotal) * 100 : 0
              return (
                <div key={label} className="flex-1">
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">{label}</span>
                    <span className="text-slate-300">{pct.toFixed(0)}%</span>
                  </div>
                  <div className="bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div className={clsx('h-2 rounded-full', cls)} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Retention Strengths */}
      {retentionStrengths.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Retention Strengths</h3>
          <div className="space-y-1">
            {retentionStrengths.map((s, i) => (
              <div key={i} className="flex items-center justify-between text-xs">
                <span className="text-emerald-300 capitalize">{(s.area ?? '').replace(/_/g, ' ')}</span>
                <span className="text-slate-500">{s.mention_count ?? ''}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Category Council */}
      {council && (council.conclusion || council.market_regime) && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-3">
            <Users className="h-4 w-4 text-emerald-400" /> Category Council
          </h3>
          <div className="flex flex-wrap gap-2 text-xs mb-2">
            {council.market_regime && <span className="px-2 py-0.5 bg-indigo-500/15 text-indigo-300 rounded">{council.market_regime}</span>}
            {council.winner && <span className="px-2 py-0.5 bg-emerald-500/15 text-emerald-300 rounded">winner: {council.winner}</span>}
            {council.loser && <span className="px-2 py-0.5 bg-red-500/15 text-red-300 rounded">loser: {council.loser}</span>}
            {council.confidence != null && council.confidence > 0 && <span className="px-2 py-0.5 bg-slate-700/50 text-slate-300 rounded">{Math.round(council.confidence * 100)}% conf.</span>}
            {council.durability && <span className="px-2 py-0.5 bg-amber-500/15 text-amber-300 rounded">{council.durability}</span>}
          </div>
          {council.conclusion && <p className="text-sm text-slate-300">{council.conclusion}</p>}
          {council.key_insights.length > 0 && (
            <ul className="mt-2 space-y-1">
              {council.key_insights.slice(0, 4).map((insight, k) => (
                <li key={k} className="text-xs text-slate-400 flex gap-2 min-w-0">
                  <span className="text-cyan-400 shrink-0">-</span>
                  <span>{insight.insight}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Case studies */}
      {studies.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-3">
            <MessageSquareQuote className="h-4 w-4 text-cyan-400" /> Customer Quotes
          </h3>
          <div className="space-y-3">
            {studies.map((s, i) => (
              <div key={i} className="border-l-2 border-cyan-500/30 pl-3">
                <blockquote className="text-xs text-slate-300 italic break-words">&quot;{s.quote}&quot;</blockquote>
                <p className="text-xs text-slate-500 mt-1">
                  {s.company ?? 'Anonymous'}
                  {s.title ? ` · ${s.title}` : ''}
                  {s.urgency ? ` · urgency ${s.urgency}` : ''}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
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
  if (s === 'very_high' || s === 'high' || s === 'strong') return 'text-red-400'
  if (s === 'medium' || s === 'moderate') return 'text-amber-400'
  if (s === 'emerging') return 'text-cyan-400'
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
  const pricingPct = typeof meta.pricing_pct === 'number' ? `${Math.round(meta.pricing_pct)}%` : null

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
              const barWidth = `${Math.min(pct, 100).toFixed(1)}%`
              return (
                <div key={i} className="flex items-center gap-3 text-xs">
                  <span className="w-32 text-slate-300 shrink-0 capitalize">{driverLabel(d.driver)}</span>
                  <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-2 rounded-full bg-cyan-500/70"
                      style={{ width: barWidth }}
                    />
                  </div>
                  <span className="w-10 text-right text-slate-400 shrink-0">{Math.round(pct)}%</span>
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
                  {b.source_archetype && <span className="text-xs text-slate-500">({b.source_archetype})</span>}
                  <span className="text-slate-500">→</span>
                  <span className="text-green-300">{b.to_vendor ?? '?'}</span>
                  {b.target_archetype && <span className="text-xs text-slate-500">({b.target_archetype})</span>}
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
                  {b.confidence_score != null && b.confidence_score > 0 && (
                    <span className="px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300 text-xs">{Math.round(b.confidence_score * 100)}% conf.</span>
                  )}
                  {b.durability && (
                    <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 text-xs">{b.durability}</span>
                  )}
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
  if (reportType === 'vendor_deep_dive') return <VendorDeepDiveDetail items={toVendorDeepDives(normalizedValue)} />
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
