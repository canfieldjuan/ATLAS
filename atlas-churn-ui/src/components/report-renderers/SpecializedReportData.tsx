import { clsx } from 'clsx'
import { MessageSquareQuote, Shield, Swords, Target, TrendingDown, Zap } from 'lucide-react'
import type { ReactNode } from 'react'
import ArchetypeBadge from '../ArchetypeBadge'
import { StructuredReportData } from './StructuredReportData'
import { normalizeReportObject, normalizeUnknown } from '../../lib/reportNormalization'
import {
  toBattleCardViewModel,
  toAccountsInMotionViewModel,
  toChallengerBriefViewModel,
  toComparisonReportViewModel,
  toVendorScorecards,
  toWeeklyChurnFeedItems,
} from '../../lib/reportViewModels'
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
} from '../../types/reportViewModels'

export const SPECIALIZED_REPORT_TYPES = [
  'challenger_brief',
  'accounts_in_motion',
  'battle_card',
  'vendor_comparison',
  'account_comparison',
  'weekly_churn_feed',
  'vendor_scorecard',
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
    <div className="space-y-6 min-w-0 [overflow-wrap:anywhere]">
      {items.map((item, index) => (
        <SectionCard
          key={`${item.vendor ?? 'feed'}-${index}`}
          title={item.vendor ?? 'Vendor'}
          icon={<TrendingDown className="h-4 w-4 text-red-400" />}
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <MetricRow label="Category" value={item.category} />
            <MetricRow label="Pressure" value={item.churn_pressure_score != null ? Math.round(item.churn_pressure_score) : '--'} color="text-red-400 font-bold" />
            <MetricRow label="Urgency" value={item.avg_urgency} />
            <MetricRow label="Reviews" value={item.total_reviews} />
          </div>
          {item.key_quote && <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 mb-3 break-words whitespace-pre-wrap">"{item.key_quote}"</blockquote>}
          {item.action_recommendation && <p className="text-sm text-cyan-300 mb-3">{item.action_recommendation}</p>}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {item.pain_breakdown.length > 0 && (
              <div>
                <p className="text-xs font-medium text-slate-400 mb-2">Pain Breakdown</p>
                <div className="space-y-1">
                  {item.pain_breakdown.slice(0, 5).map((pain, painIndex) => (
                    <MetricRow key={painIndex} label={pain.category ?? pain.feature ?? ''} value={pain.count ?? pain.mentions} />
                  ))}
                </div>
              </div>
            )}
            {item.top_displacement_targets.length > 0 && (
              <div>
                <p className="text-xs font-medium text-slate-400 mb-2">Top Displacement Targets</p>
                <div className="space-y-1">
                  {item.top_displacement_targets.slice(0, 5).map((target: CompetitorDifferentiatorViewModel, targetIndex: number) => (
                    <MetricRow key={targetIndex} label={target.competitor ?? ''} value={target.mentions ?? target.count} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </SectionCard>
      ))}
    </div>
  )
}

function VendorScorecardDetail({ items }: { items: VendorScorecardViewModel[] }) {
  return (
    <div className="space-y-6 min-w-0 [overflow-wrap:anywhere]">
      {items.map((item, index) => (
        <SectionCard
          key={`${item.vendor ?? 'scorecard'}-${index}`}
          title={item.vendor ?? 'Vendor'}
          icon={<Shield className="h-4 w-4 text-cyan-400" />}
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricRow label="Reviews" value={item.total_reviews} />
            <MetricRow label="Signal Density" value={item.churn_signal_density} />
            <MetricRow label="Avg Urgency" value={item.avg_urgency} />
            <MetricRow label="Recommend Ratio" value={item.recommend_ratio} />
            <MetricRow label="Top Pain" value={item.top_pain} />
            <MetricRow label="Competitor Threat" value={item.top_competitor_threat} />
            <MetricRow label="Trend" value={item.trend} />
            <MetricRow label="Sentiment" value={item.sentiment_direction} />
          </div>
        </SectionCard>
      ))}
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
  return null
}
