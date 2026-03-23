import { useState, useEffect } from 'react'
import { FileText, X, Shield, Target, Swords, MessageSquareQuote, TrendingDown, Zap, Users, BarChart3 } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import FilterBar, { FilterSelect } from '../../components/FilterBar'
import { fetchReports, fetchReportDetail, type B2BReport, type B2BReportDetail } from '../../api/b2bClient'

/* ------------------------------------------------------------------ */
/*  Shared tiny helpers                                                */
/* ------------------------------------------------------------------ */

type AnyData = Record<string, any>

function Section({ title, icon, children }: { title: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="mt-5 min-w-0">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-cyan-400 uppercase tracking-wide mb-2 min-w-0 break-words">
        {icon}{title}
      </h3>
      <div className="space-y-2 min-w-0">{children}</div>
    </div>
  )
}

function Metric({ label, value, color }: { label: string; value: string | number | null | undefined; color?: string }) {
  const display = value == null || value === '' ? '--' : value
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3 text-sm py-0.5 items-start">
      <span className="text-slate-400 min-w-0 break-words">{label}</span>
      <span className={`min-w-0 break-words text-right ${display === '--' ? 'text-slate-600' : (color || 'text-white font-medium')}`}>{display}</span>
    </div>
  )
}

function QuoteBlock({ text }: { text: string }) {
  return (
    <blockquote className="border-l-2 border-cyan-700/60 pl-3 py-1 text-xs text-slate-300 italic leading-relaxed break-words whitespace-pre-wrap">
      {text.slice(0, 300)}
    </blockquote>
  )
}

function MiniTable({ headers, rows }: { headers: string[]; rows: (string | number)[][] }) {
  if (!rows.length) return null
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs table-auto">
        <thead>
          <tr className="border-b border-slate-700/50">
            {headers.map((h, i) => (
              <th key={i} className="text-left text-slate-500 font-medium py-1.5 px-2 align-top break-words">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-b border-slate-800/30 hover:bg-slate-800/30">
              {row.map((cell, ci) => (
                <td key={ci} className="text-slate-300 py-1.5 px-2 align-top break-words">{cell ?? '--'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Badge({ text, color }: { text: string; color: string }) {
  return <span className={`inline-block text-xs px-2 py-0.5 rounded-full font-medium ${color}`}>{text}</span>
}

function qualityBadge(status: string | null | undefined) {
  const key = (status || '').toLowerCase()
  if (key === 'sales_ready') return <Badge text="Sales Ready" color="bg-emerald-900/40 text-emerald-300" />
  if (key === 'needs_review') return <Badge text="Needs Review" color="bg-amber-900/40 text-amber-300" />
  if (key === 'deterministic_fallback') return <Badge text="Fallback" color="bg-rose-900/40 text-rose-300" />
  return null
}

function riskColor(level: string | undefined) {
  if (!level) return 'text-slate-400'
  const l = level.toLowerCase()
  if (l === 'high' || l === 'critical') return 'text-red-400'
  if (l === 'medium') return 'text-amber-400'
  return 'text-green-400'
}

function pct(v: number | null | undefined): string {
  if (v == null) return '--'
  return `${(v * 100).toFixed(1)}%`
}

function fmt(v: string | undefined | null): string {
  if (!v) return '--'
  return v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

/* ------------------------------------------------------------------ */
/*  Challenger Brief renderer                                          */
/* ------------------------------------------------------------------ */

function ChallengerBriefView({ d }: { d: AnyData }) {
  const disp = (d.displacement_summary || {}) as AnyData
  const inc = (d.incumbent_profile || {}) as AnyData
  const adv = (d.challenger_advantage || {}) as AnyData
  const h2h = (d.head_to_head || {}) as AnyData
  const targets = (d.target_accounts || []) as AnyData[]
  const playbook = (d.sales_playbook || {}) as AnyData
  const integ = (d.integration_comparison || {}) as AnyData
  const sources = (d.data_sources || {}) as AnyData

  return (
    <>
      {/* Header */}
      <div className="flex items-center gap-3 mb-1">
        <Swords className="h-5 w-5 text-cyan-400" />
        <span className="text-lg font-bold text-white">{d.challenger} vs {d.incumbent}</span>
      </div>
      {d.category && <span className="text-xs text-slate-500">{d.category}</span>}

      {/* Data sources pills */}
      <div className="flex flex-wrap gap-1.5 mt-3">
        {Object.entries(sources).map(([k, v]) => (
          <Badge key={k} text={fmt(k)} color={v ? 'bg-cyan-900/40 text-cyan-300' : 'bg-slate-800 text-slate-600'} />
        ))}
      </div>

      {/* Displacement Signal */}
      <Section title="Displacement Signal" icon={<TrendingDown className="h-4 w-4" />}>
        <div className="bg-slate-900/40 rounded-lg p-3 space-y-1">
          <Metric label="Total Mentions" value={disp.total_mentions} color={disp.total_mentions >= 50 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />
          <Metric label="Signal Strength" value={fmt(disp.signal_strength)} />
          <Metric label="Confidence" value={typeof disp.confidence_score === 'number' ? pct(disp.confidence_score) : disp.confidence_score} />
          <Metric label="Primary Driver" value={fmt(disp.primary_driver)} />
        </div>
        {disp.source_distribution && typeof disp.source_distribution === 'object' && Object.keys(disp.source_distribution).length > 0 && (
          <div className="text-xs text-slate-500 mt-1">
            Sources: {Object.entries(disp.source_distribution as Record<string, number>)
              .sort(([, a], [, b]) => b - a)
              .map(([src, cnt]) => `${src}: ${cnt}`)
              .join(', ')}
          </div>
        )}
        {disp.key_quote && <QuoteBlock text={disp.key_quote} />}
      </Section>

      {/* Incumbent Profile */}
      {Object.keys(inc).length > 0 && (
        <Section title={`Incumbent: ${d.incumbent}`} icon={<Shield className="h-4 w-4" />}>
          <div className="bg-slate-900/40 rounded-lg p-3 space-y-1">
            <Metric label="Archetype" value={fmt(inc.archetype)} />
            {inc.archetype_confidence != null && <Metric label="Archetype Confidence" value={pct(inc.archetype_confidence)} />}
            <Metric label="Risk Level" value={fmt(inc.risk_level)} color={riskColor(inc.risk_level)} />
            {inc.churn_pressure_score != null && <Metric label="Churn Pressure" value={`${Number(inc.churn_pressure_score).toFixed(0)}/100`} color={Number(inc.churn_pressure_score) >= 60 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />}
            <Metric label="Price Complaint Rate" value={pct(inc.price_complaint_rate)} />
            <Metric label="DM Churn Rate" value={pct(inc.dm_churn_rate)} />
            <Metric label="Sentiment Trend" value={fmt(inc.sentiment_direction)} />
          </div>

          {Array.isArray(inc.top_weaknesses) && inc.top_weaknesses.length > 0 && (
            <MiniTable
              headers={['Weakness', 'Evidence']}
              rows={inc.top_weaknesses.slice(0, 8).map((w: AnyData) => [
                fmt(w.area || w.weakness || w.name || ''),
                w.count ?? w.evidence_count ?? '',
              ])}
            />
          )}

          {Array.isArray(inc.top_pain_quotes) && inc.top_pain_quotes.length > 0 && (
            <div className="space-y-2 mt-2">
              {inc.top_pain_quotes.slice(0, 3).map((q: any, i: number) => (
                <QuoteBlock key={i} text={typeof q === 'string' ? q : q?.quote || ''} />
              ))}
            </div>
          )}
        </Section>
      )}

      {/* Challenger Advantage */}
      {(Array.isArray(adv.strengths) && adv.strengths.length > 0 || Array.isArray(adv.weakness_coverage) && adv.weakness_coverage.length > 0 || adv.profile_summary) && (
        <Section title={`Challenger: ${d.challenger}`} icon={<Zap className="h-4 w-4" />}>
          {adv.profile_summary && <p className="text-xs text-slate-300">{adv.profile_summary}</p>}

          {Array.isArray(adv.strengths) && adv.strengths.length > 0 && (
            <MiniTable
              headers={['Strength', 'Evidence']}
              rows={adv.strengths.slice(0, 8).map((s: AnyData) => [
                fmt(s.area || s.name || ''),
                s.evidence_count ?? s.mentions ?? '',
              ])}
            />
          )}

          {Array.isArray(adv.weakness_coverage) && adv.weakness_coverage.length > 0 && (
            <>
              <p className="text-xs font-medium text-slate-400 mt-2">Weakness Coverage</p>
              <MiniTable
                headers={['Incumbent Weakness', 'Match Quality']}
                rows={adv.weakness_coverage.slice(0, 8).map((c: AnyData) => [
                  fmt(c.incumbent_weakness || ''),
                  c.match_quality || '',
                ])}
              />
            </>
          )}

          {Array.isArray(adv.commonly_switched_from) && adv.commonly_switched_from.length > 0 && (
            <>
              <p className="text-xs font-medium text-slate-400 mt-2">Commonly Switched From</p>
              <MiniTable
                headers={['Vendor', 'Count', 'Reason']}
                rows={adv.commonly_switched_from.slice(0, 6).map((sf: AnyData) => [
                  typeof sf === 'string' ? sf : sf.vendor || '',
                  typeof sf === 'string' ? '' : sf.count ?? '',
                  typeof sf === 'string' ? '' : (sf.top_reason || '').slice(0, 60),
                ])}
              />
            </>
          )}
        </Section>
      )}

      {/* Head to Head */}
      {Object.keys(h2h).length > 0 && (h2h.conclusion || h2h.winner) && (
        <Section title="Head to Head" icon={<Swords className="h-4 w-4" />}>
          <div className="bg-slate-900/40 rounded-lg p-3 space-y-1">
            {h2h.winner && <Metric label="Winner" value={h2h.winner} color="text-cyan-400 font-bold" />}
            {h2h.confidence != null && <Metric label="Confidence" value={pct(h2h.confidence)} />}
            {h2h.durability && <Metric label="Durability" value={h2h.durability} />}
            {h2h.synthesized && <span className="text-[10px] text-slate-600">(synthesized from displacement data)</span>}
          </div>
          {h2h.conclusion && <p className="text-xs text-slate-300 mt-1">{h2h.conclusion}</p>}
          {Array.isArray(h2h.key_insights) && h2h.key_insights.length > 0 && (
            <ul className="list-disc list-inside text-xs text-slate-400 space-y-1 mt-1">
              {h2h.key_insights.slice(0, 5).map((ins: any, i: number) => {
                const text = typeof ins === 'string' ? ins : ins?.insight || ''
                const evidence = typeof ins === 'object' ? ins?.evidence : ''
                return (
                  <li key={i}>
                    {text}
                    {evidence && <span className="text-slate-600 ml-1">({evidence})</span>}
                  </li>
                )
              })}
            </ul>
          )}
        </Section>
      )}

      {/* Target Accounts */}
      {targets.length > 0 && (
        <Section title={`Target Accounts (${d.total_target_accounts ?? targets.length} total, ${d.accounts_considering_challenger ?? 0} considering ${d.challenger})`} icon={<Target className="h-4 w-4" />}>
          <MiniTable
            headers={['Company', 'Score', 'Stage', 'Urg', 'Industry', 'Chall?']}
            rows={targets.slice(0, 15).map((t: AnyData) => [
              t.company || '',
              t.opportunity_score ?? '',
              fmt(t.buying_stage || ''),
              typeof t.urgency === 'number' ? t.urgency.toFixed(0) : '',
              (t.industry || '').slice(0, 20),
              t.considers_challenger ? 'Y' : '',
            ])}
          />
        </Section>
      )}

      {/* Sales Playbook */}
      {Object.keys(playbook).length > 0 && (
        <Section title="Sales Playbook" icon={<MessageSquareQuote className="h-4 w-4" />}>
          <PlaybookView playbook={playbook} />
        </Section>
      )}

      {/* Integration Comparison */}
      {(Array.isArray(integ.shared) && integ.shared.length > 0 || Array.isArray(integ.challenger_exclusive) && integ.challenger_exclusive.length > 0) && (
        <Section title="Integration Comparison">
          {Array.isArray(integ.shared) && integ.shared.length > 0 && (
            <Metric label="Shared" value={integ.shared.slice(0, 10).join(', ')} />
          )}
          {Array.isArray(integ.challenger_exclusive) && integ.challenger_exclusive.length > 0 && (
            <Metric label={`${d.challenger} Exclusive`} value={integ.challenger_exclusive.slice(0, 10).join(', ')} />
          )}
          {Array.isArray(integ.incumbent_exclusive) && integ.incumbent_exclusive.length > 0 && (
            <Metric label={`${d.incumbent} Exclusive`} value={integ.incumbent_exclusive.slice(0, 10).join(', ')} />
          )}
        </Section>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Shared Playbook sub-renderer                                       */
/* ------------------------------------------------------------------ */

function PlaybookView({ playbook }: { playbook: AnyData }) {
  const discovery = Array.isArray(playbook.discovery_questions) ? playbook.discovery_questions : []
  const landmines = Array.isArray(playbook.landmine_questions) ? playbook.landmine_questions : []
  const objections = Array.isArray(playbook.objection_handlers) ? playbook.objection_handlers : []
  const talk = playbook.talk_track

  return (
    <div className="space-y-3">
      {discovery.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-400 mb-1">Discovery Questions</p>
          <ul className="list-disc list-inside text-xs text-slate-300 space-y-0.5">
            {discovery.slice(0, 5).map((q: string, i: number) => <li key={i}>{q}</li>)}
          </ul>
        </div>
      )}
      {landmines.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-400 mb-1">Landmine Questions</p>
          <ul className="list-disc list-inside text-xs text-slate-300 space-y-0.5">
            {landmines.slice(0, 3).map((q: string, i: number) => <li key={i}>{q}</li>)}
          </ul>
        </div>
      )}
      {objections.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-400 mb-1">Objection Handlers</p>
          {objections.slice(0, 3).map((obj: AnyData, i: number) => (
            <div key={i} className="bg-slate-900/30 rounded p-2 mb-2">
              <p className="text-xs text-white italic">"{obj.objection}"</p>
              {obj.pivot && <p className="text-xs text-slate-300 mt-1">{obj.pivot}</p>}
              {obj.proof_point && <p className="text-[10px] text-slate-500 mt-0.5">{obj.proof_point}</p>}
            </div>
          ))}
        </div>
      )}
      {talk && typeof talk === 'object' && !Array.isArray(talk) && (
        <div>
          <p className="text-xs font-medium text-slate-400 mb-1">Talk Track</p>
          {(['opening', 'mid_call_pivot', 'closing'] as const).map(phase => {
            const text = (talk as AnyData)[phase]
            if (!text) return null
            return (
              <div key={phase} className="mb-1.5">
                <span className="text-[10px] text-slate-500 uppercase">{fmt(phase)}</span>
                <p className="text-xs text-slate-300">{text}</p>
              </div>
            )
          })}
        </div>
      )}
      {talk && typeof talk === 'string' && <p className="text-xs text-slate-300">{talk}</p>}
      {Array.isArray(playbook.recommended_plays) && playbook.recommended_plays.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-400 mb-1">Recommended Plays</p>
          {playbook.recommended_plays.slice(0, 3).map((play: AnyData, i: number) => (
            <div key={i} className="bg-slate-900/30 rounded p-2 mb-1.5">
              <p className="text-xs text-white font-medium">{play.play}</p>
              {play.target_segment && <p className="text-[10px] text-slate-500">Target: {play.target_segment}</p>}
              {play.key_message && <p className="text-[10px] text-slate-400">{play.key_message}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Accounts In Motion renderer                                        */
/* ------------------------------------------------------------------ */

function AccountsInMotionView({ d }: { d: AnyData }) {
  const pricing = (d.pricing_pressure || {}) as AnyData
  const gaps = Array.isArray(d.feature_gaps) ? d.feature_gaps : []
  const xvc = (d.cross_vendor_context || {}) as AnyData
  const accounts = Array.isArray(d.accounts) ? d.accounts : []

  return (
    <>
      <div className="flex items-center gap-3 mb-1">
        <Users className="h-5 w-5 text-cyan-400" />
        <span className="text-lg font-bold text-white">Accounts In Motion: {d.vendor}</span>
      </div>
      {d.category && <span className="text-xs text-slate-500">{d.category}</span>}

      <div className="bg-slate-900/40 rounded-lg p-3 space-y-1 mt-3">
        <Metric label="Total Accounts" value={d.total_accounts_in_motion} color="text-cyan-400 font-bold" />
        <Metric label="Archetype" value={fmt(d.archetype)} />
        {d.archetype_confidence != null && <Metric label="Confidence" value={pct(d.archetype_confidence)} />}
      </div>

      {/* Pricing */}
      {(pricing.price_complaint_rate != null || pricing.price_increase_rate != null) && (
        <Section title="Pricing Pressure" icon={<BarChart3 className="h-4 w-4" />}>
          <div className="bg-slate-900/40 rounded-lg p-3 space-y-1">
            <Metric label="Price Complaint Rate" value={pct(pricing.price_complaint_rate)} />
            <Metric label="Price Increase Rate" value={pct(pricing.price_increase_rate)} />
            {pricing.avg_seat_count != null && <Metric label="Avg Seat Count" value={Math.round(pricing.avg_seat_count)} />}
          </div>
        </Section>
      )}

      {/* Feature Gaps */}
      {gaps.length > 0 && (
        <Section title="Top Feature Gaps">
          <MiniTable
            headers={['Feature', 'Mentions']}
            rows={gaps.slice(0, 10).map((g: AnyData) => [
              fmt(g.feature || ''),
              g.mentions ?? '',
            ])}
          />
        </Section>
      )}

      {/* Cross-Vendor Context */}
      {(xvc.top_destination || xvc.battle_conclusion || xvc.market_regime) && (
        <Section title="Competitive Context">
          <div className="bg-slate-900/40 rounded-lg p-3 space-y-1">
            <Metric label="Top Destination" value={xvc.top_destination} />
            <Metric label="Market Regime" value={fmt(xvc.market_regime)} />
          </div>
          {xvc.battle_conclusion && <p className="text-xs text-slate-300 mt-1">{xvc.battle_conclusion}</p>}
        </Section>
      )}

      {/* Accounts table */}
      {accounts.length > 0 && (
        <Section title={`Prospecting List (${accounts.length} accounts)`} icon={<Target className="h-4 w-4" />}>
          <MiniTable
            headers={['Company', 'Score', 'Stage', 'Urg', 'Industry', 'Domain']}
            rows={accounts.slice(0, 25).map((a: AnyData) => [
              (a.company || '').slice(0, 20),
              a.opportunity_score ?? '',
              fmt(a.buying_stage || ''),
              typeof a.urgency === 'number' ? a.urgency.toFixed(0) : '',
              (a.industry || '').slice(0, 18),
              (a.domain || '').slice(0, 24),
            ])}
          />

          {/* Key quotes from accounts */}
          {accounts.filter((a: AnyData) => a.top_quote).length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-slate-400 mb-1">Key Quotes</p>
              {accounts.filter((a: AnyData) => a.top_quote).slice(0, 4).map((a: AnyData, i: number) => (
                <div key={i} className="mb-2">
                  <span className="text-[10px] text-slate-500">{a.company}{a.urgency ? ` (urgency: ${a.urgency})` : ''}</span>
                  <QuoteBlock text={a.top_quote} />
                </div>
              ))}
            </div>
          )}
        </Section>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Battle Card renderer                                               */
/* ------------------------------------------------------------------ */

function BattleCardView({ d }: { d: AnyData }) {
  const weaknesses = Array.isArray(d.vendor_weaknesses) ? d.vendor_weaknesses : []
  const quotes = Array.isArray(d.customer_pain_quotes) ? d.customer_pain_quotes : []
  const diffs = Array.isArray(d.competitor_differentiators) ? d.competitor_differentiators : []
  const objections = Array.isArray(d.objection_handlers) ? d.objection_handlers : []
  const plays = Array.isArray(d.recommended_plays) ? d.recommended_plays : []
  const qualityStatus = d.quality_status || d?.battle_card_quality?.status
  const qualityScore = d?.battle_card_quality?.score

  return (
    <>
      <div className="flex items-center gap-3 mb-1">
        <Shield className="h-5 w-5 text-cyan-400" />
        <span className="text-lg font-bold text-white">Battle Card: {d.vendor || 'Unknown'}</span>
        {qualityBadge(qualityStatus)}
      </div>

      <div className="bg-slate-900/40 rounded-lg p-3 space-y-1 mt-3">
        <Metric label="Churn Pressure" value={d.churn_pressure_score != null ? `${Number(d.churn_pressure_score).toFixed(0)}/100` : null} color={Number(d.churn_pressure_score || 0) >= 60 ? 'text-red-400 font-bold' : 'text-amber-400 font-bold'} />
        <Metric label="Risk Level" value={fmt(d.risk_level)} color={riskColor(d.risk_level)} />
        <Metric label="Total Reviews" value={d.total_reviews} />
        <Metric label="Confidence" value={d.confidence} />
        <Metric label="Quality Score" value={qualityScore ?? '--'} />
      </div>

      {weaknesses.length > 0 && (
        <Section title="Vendor Weaknesses">
          <MiniTable
            headers={['Weakness', 'Evidence', 'Source']}
            rows={weaknesses.slice(0, 8).map((w: AnyData) => [
              fmt(w.area || ''),
              w.evidence_count ?? w.count ?? '',
              w.source || '',
            ])}
          />
        </Section>
      )}

      {quotes.length > 0 && (
        <Section title="Customer Pain Points">
          {quotes.slice(0, 4).map((q: any, i: number) => (
            <QuoteBlock key={i} text={typeof q === 'string' ? q : q?.quote || ''} />
          ))}
        </Section>
      )}

      {diffs.length > 0 && (
        <Section title="Competitor Differentiators">
          <MiniTable
            headers={['Competitor', 'Mentions', 'Solves', 'Driver']}
            rows={diffs.slice(0, 6).map((dd: AnyData) => [
              dd.competitor || '',
              dd.mentions ?? '',
              (dd.solves_weakness || '').slice(0, 30),
              dd.primary_driver || '',
            ])}
          />
        </Section>
      )}

      {objections.length > 0 && (
        <Section title="Objection Handlers">
          {objections.slice(0, 3).map((obj: AnyData, i: number) => (
            <div key={i} className="bg-slate-900/30 rounded p-2 mb-2">
              <p className="text-xs text-white italic">"{obj.objection}"</p>
              {obj.pivot && <p className="text-xs text-slate-300 mt-1">{obj.pivot}</p>}
            </div>
          ))}
        </Section>
      )}

      {plays.length > 0 && (
        <Section title="Recommended Plays">
          {plays.slice(0, 3).map((play: AnyData, i: number) => (
            <div key={i} className="bg-slate-900/30 rounded p-2 mb-1.5">
              <p className="text-xs text-white font-medium">{play.play}</p>
              {play.target_segment && <p className="text-[10px] text-slate-500">Target: {play.target_segment}</p>}
              {play.key_message && <p className="text-[10px] text-slate-400">{play.key_message}</p>}
            </div>
          ))}
        </Section>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Generic fallback renderer (still better than raw JSON)             */
/* ------------------------------------------------------------------ */

function GenericReportView({ d }: { d: AnyData }) {
  return (
    <div className="space-y-2">
      {Object.entries(d).map(([key, val]) => {
        if (key.startsWith('_')) return null
        if (val == null) return null
        if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
          return <Metric key={key} label={fmt(key)} value={String(val)} />
        }
        if (Array.isArray(val) && val.length > 0) {
          if (typeof val[0] === 'object' && val[0] !== null) {
            const headers = Object.keys(val[0]).slice(0, 5)
            return (
              <Section key={key} title={fmt(key)}>
                <MiniTable
                  headers={headers.map(fmt)}
                  rows={val.slice(0, 15).map(item =>
                    headers.map(h => String(item?.[h] ?? '').slice(0, 40))
                  )}
                />
              </Section>
            )
          }
          return (
            <Section key={key} title={fmt(key)}>
              <ul className="list-disc list-inside text-xs text-slate-300 space-y-0.5">
                {val.slice(0, 10).map((item, i) => <li key={i}>{String(item).slice(0, 200)}</li>)}
              </ul>
            </Section>
          )
        }
        if (typeof val === 'object') {
          return (
            <Section key={key} title={fmt(key)}>
              {Object.entries(val).map(([k2, v2]) => (
                <Metric key={k2} label={fmt(k2)} value={v2 != null ? String(v2).slice(0, 100) : '--'} />
              ))}
            </Section>
          )
        }
        return null
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Report content dispatcher                                          */
/* ------------------------------------------------------------------ */

function ReportContent({ reportType, data }: { reportType: string; data: unknown }) {
  const d = (data && typeof data === 'object' ? data : {}) as AnyData
  switch (reportType) {
    case 'challenger_brief':
    case 'challenger_intel':
      return <ChallengerBriefView d={d} />
    case 'accounts_in_motion':
      return <AccountsInMotionView d={d} />
    case 'battle_card':
      return <BattleCardView d={d} />
    default:
      return <GenericReportView d={d} />
  }
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function B2BReports() {
  const [reports, setReports] = useState<B2BReport[]>([])
  const [loading, setLoading] = useState(true)
  const [reportType, setReportType] = useState('')
  const [selected, setSelected] = useState<B2BReportDetail | null>(null)

  const handleReportTypeChange = (value: string) => {
    setLoading(true)
    setReportType(value)
  }

  useEffect(() => {
    fetchReports({ report_type: reportType || undefined, limit: 50 })
      .then(r => setReports(r.reports))
      .catch(() => setReports([]))
      .finally(() => setLoading(false))
  }, [reportType])

  const handleRowClick = async (r: B2BReport) => {
    try {
      const detail = await fetchReportDetail(r.id)
      setSelected(detail)
    } catch {
      // ignore
    }
  }

  const columns: Column<B2BReport>[] = [
    {
      key: 'type',
      header: 'Type',
      render: r => <span className="text-cyan-400 text-xs font-medium">{r.report_type}</span>,
    },
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-white">{r.vendor_filter || 'All'}</span>,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: r => (
        <span className="text-slate-400 line-clamp-2 max-w-md">
          {r.executive_summary || '--'}
        </span>
      ),
    },
    {
      key: 'date',
      header: 'Date',
      sortable: true,
      sortValue: r => r.report_date || '',
      render: r => <span className="text-slate-500 text-xs">{r.report_date?.split('T')[0] || '--'}</span>,
    },
    {
      key: 'status',
      header: 'Status',
      render: r => {
        if (r.report_type === 'battle_card') {
          const badge = qualityBadge(r.quality_status)
          if (badge) return badge
        }
        return (
          <span className={r.status === 'published' ? 'text-green-400 text-xs' : 'text-slate-500 text-xs'}>
            {r.status || 'draft'}
          </span>
        )
      },
    },
  ]

  const reportTypes = [
    { value: 'weekly_churn_feed', label: 'Weekly Churn Feed' },
    { value: 'vendor_scorecard', label: 'Vendor Scorecard' },
    { value: 'displacement_report', label: 'Displacement Report' },
    { value: 'category_overview', label: 'Category Overview' },
    { value: 'exploratory_overview', label: 'Exploratory Overview' },
    { value: 'vendor_comparison', label: 'Vendor Comparison' },
    { value: 'account_comparison', label: 'Account Comparison' },
    { value: 'account_deep_dive', label: 'Account Deep Dive' },
    { value: 'vendor_retention', label: 'Vendor Retention' },
    { value: 'challenger_brief', label: 'Challenger Brief' },
    { value: 'challenger_intel', label: 'Challenger Intel' },
    { value: 'battle_card', label: 'Battle Card' },
    { value: 'accounts_in_motion', label: 'Accounts In Motion' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <FileText className="h-6 w-6 text-cyan-400" />
        <h1 className="text-2xl font-bold text-white">Intelligence Reports</h1>
      </div>

      <FilterBar>
        <FilterSelect
          label="Report Type"
          value={reportType}
          onChange={handleReportTypeChange}
          options={reportTypes}
          placeholder="All types"
        />
      </FilterBar>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={reports}
          onRowClick={handleRowClick}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No reports available for your tracked vendors"
        />
      </div>

      {/* Detail modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60" onClick={() => setSelected(null)}>
          <div
            className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-6xl w-full max-h-[85vh] overflow-y-auto overflow-x-hidden [overflow-wrap:anywhere]"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-white">{fmt(selected.report_type)}</h2>
                <div className="flex gap-3 text-xs text-slate-500 mt-0.5">
                  {selected.vendor_filter && <span>Vendor: {selected.vendor_filter}</span>}
                  {selected.category_filter && <span>Category: {selected.category_filter}</span>}
                  <span>{selected.report_date?.split('T')[0]}</span>
                </div>
              </div>
              <button onClick={() => setSelected(null)} className="text-slate-400 hover:text-white p-1">
                <X className="h-5 w-5" />
              </button>
            </div>

            {selected.executive_summary && (
              <p className="text-sm text-slate-300 mb-4 pb-3 border-b border-slate-700/50">
                {selected.executive_summary}
              </p>
            )}

            <ReportContent reportType={selected.report_type} data={selected.intelligence_data} />
          </div>
        </div>
      )}
    </div>
  )
}
