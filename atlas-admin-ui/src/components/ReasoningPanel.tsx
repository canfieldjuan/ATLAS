import { Brain, RefreshCcw, FileText, Swords, ArrowRight } from 'lucide-react'
import type { RecentCall, WorkflowEntry } from '../types'
import { fmtCost, fmtTokens } from '../utils'

type ReasoningBucket = 'reason' | 'reconstitute' | 'report' | 'battle_card_copy' | 'other'

type ReasoningClass = {
  bucket: ReasoningBucket
  label: string
  tone: string
  description: string
}

function classifySpan(spanName: string): ReasoningClass {
  if (spanName === 'reasoning.stratified.reason') {
    return {
      bucket: 'reason',
      label: 'Full Reason',
      tone: 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300',
      description: 'Fresh stratified reasoning'
    }
  }
  if (spanName === 'reasoning.stratified.reconstitute') {
    return {
      bucket: 'reconstitute',
      label: 'Reconstitute',
      tone: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
      description: 'Patch prior conclusion from delta'
    }
  }
  if (spanName === 'b2b.churn_intelligence.battle_card_sales_copy') {
    return {
      bucket: 'battle_card_copy',
      label: 'Battle Card Copy',
      tone: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
      description: 'Sales copy over deterministic card'
    }
  }
  if (
    spanName === 'b2b.churn_intelligence.exploratory_overview' ||
    spanName === 'b2b.churn_intelligence.scorecard_narrative' ||
    spanName === 'b2b.churn_intelligence.executive_summary'
  ) {
    return {
      bucket: 'report',
      label: 'Report Synthesis',
      tone: 'border-violet-500/30 bg-violet-500/10 text-violet-300',
      description: 'Downstream narrative and summary generation'
    }
  }
  return {
    bucket: 'other',
    label: 'Other',
    tone: 'border-slate-700 bg-slate-800/50 text-slate-400',
    description: 'Other LLM activity'
  }
}

function bucketIcon(bucket: ReasoningBucket) {
  if (bucket === 'reason') return Brain
  if (bucket === 'reconstitute') return RefreshCcw
  if (bucket === 'battle_card_copy') return Swords
  return FileText
}

function bucketSummary(workflows: WorkflowEntry[], bucket: ReasoningBucket) {
  return workflows.reduce(
    (acc, workflow) => {
      if (classifySpan(workflow.workflow).bucket !== bucket) return acc
      acc.calls += workflow.calls
      acc.cost += workflow.cost_usd
      acc.tokens += workflow.total_tokens
      return acc
    },
    { calls: 0, cost: 0, tokens: 0 }
  )
}

export default function ReasoningPanel({
  workflows,
  recent,
}: {
  workflows: WorkflowEntry[]
  recent: RecentCall[]
}) {
  const fullReason = bucketSummary(workflows, 'reason')
  const reconstitute = bucketSummary(workflows, 'reconstitute')
  const report = bucketSummary(workflows, 'report')
  const battleCardCopy = bucketSummary(workflows, 'battle_card_copy')
  const totalVisibleCost = fullReason.cost + reconstitute.cost + report.cost + battleCardCopy.cost
  const reasoningCost = fullReason.cost + reconstitute.cost
  const reasoningShare = totalVisibleCost > 0 ? (reasoningCost / totalVisibleCost) * 100 : 0

  const relevantWorkflows = workflows
    .filter((workflow) => classifySpan(workflow.workflow).bucket !== 'other')
    .sort((a, b) => b.cost_usd - a.cost_usd)

  const relevantRecent = recent
    .filter((call) => classifySpan(call.span_name).bucket !== 'other')
    .slice(0, 8)

  const cards = [
    { title: 'Full Reason', stats: fullReason, bucket: 'reason' as const },
    { title: 'Reconstitute', stats: reconstitute, bucket: 'reconstitute' as const },
    { title: 'Report Synthesis', stats: report, bucket: 'report' as const },
    { title: 'Battle Card Copy', stats: battleCardCopy, bucket: 'battle_card_copy' as const },
  ]

  return (
    <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '380ms' }}>
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            Reasoning Activity
          </h2>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            Separates true stratified reasoning from downstream narrative generation. Recall hits and deterministic builders are intentionally absent because they do not create LLM cost records.
          </p>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-3 text-right">
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-600">Reasoning Share</div>
          <div className="mt-1 text-xl font-semibold text-slate-100">{reasoningShare.toFixed(0)}%</div>
          <div className="text-xs text-slate-500">
            {fmtCost(reasoningCost)} of {fmtCost(totalVisibleCost)}
          </div>
        </div>
      </div>

      <div className="mb-6 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => {
          const kind = classifySpan(
            card.bucket === 'reason'
              ? 'reasoning.stratified.reason'
              : card.bucket === 'reconstitute'
                ? 'reasoning.stratified.reconstitute'
                : card.bucket === 'battle_card_copy'
                  ? 'b2b.churn_intelligence.battle_card_sales_copy'
                  : 'b2b.churn_intelligence.executive_summary'
          )
          const Icon = bucketIcon(card.bucket)
          return (
            <div key={card.title} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-medium text-slate-300">{card.title}</div>
                  <div className="mt-1 text-[11px] text-slate-500">{kind.description}</div>
                </div>
                <div className={`rounded-lg border p-2 ${kind.tone}`}>
                  <Icon className="h-4 w-4" />
                </div>
              </div>
              <div className="mt-4 flex items-end justify-between gap-3">
                <div>
                  <div className="text-2xl font-semibold text-slate-100">{card.stats.calls}</div>
                  <div className="text-[11px] uppercase tracking-wider text-slate-600">Calls</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm text-cyan-400">{fmtCost(card.stats.cost)}</div>
                  <div className="text-[11px] text-slate-500">{fmtTokens(card.stats.tokens)} tokens</div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.9fr]">
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950/50">
          <div className="border-b border-slate-800 px-4 py-3 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            Costed Phases
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] uppercase tracking-widest text-slate-600">
                  <th className="px-4 py-3 font-semibold">Type</th>
                  <th className="px-4 py-3 font-semibold">Workflow</th>
                  <th className="px-4 py-3 text-right font-semibold">Calls</th>
                  <th className="px-4 py-3 text-right font-semibold">Tokens</th>
                  <th className="px-4 py-3 text-right font-semibold">Cost</th>
                </tr>
              </thead>
              <tbody>
                {relevantWorkflows.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-10 text-center text-sm text-slate-600">
                      No reasoning-related LLM activity recorded yet
                    </td>
                  </tr>
                )}
                {relevantWorkflows.map((workflow) => {
                  const kind = classifySpan(workflow.workflow)
                  return (
                    <tr
                      key={workflow.workflow}
                      className="border-b border-slate-800/40 transition-colors hover:bg-slate-800/20"
                    >
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider ${kind.tone}`}>
                          {kind.label}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-mono text-xs text-slate-300">{workflow.workflow}</div>
                        <div className="mt-1 text-[11px] text-slate-500">{kind.description}</div>
                      </td>
                      <td className="px-4 py-3 text-right text-slate-400">{workflow.calls}</td>
                      <td className="px-4 py-3 text-right text-slate-400">{fmtTokens(workflow.total_tokens)}</td>
                      <td className="px-4 py-3 text-right font-mono text-cyan-400">{fmtCost(workflow.cost_usd)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/50">
          <div className="border-b border-slate-800 px-4 py-3 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            Recent Sequence
          </div>
          <div className="space-y-1 p-3">
            {relevantRecent.length === 0 && (
              <div className="px-2 py-8 text-center text-sm text-slate-600">
                No recent reasoning-related calls
              </div>
            )}
            {relevantRecent.map((call, index) => {
              const kind = classifySpan(call.span_name)
              return (
                <div
                  key={`${call.span_name}-${call.created_at}-${index}`}
                  className="flex items-start gap-3 rounded-lg border border-transparent px-2 py-2 transition-colors hover:border-slate-800 hover:bg-slate-800/20"
                >
                  <span className={`mt-0.5 inline-flex shrink-0 items-center rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-wider ${kind.tone}`}>
                    {kind.label}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm text-slate-200">{call.detail || call.title || call.span_name}</div>
                    <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-500">
                      <span className="truncate font-mono text-slate-600">{call.span_name}</span>
                      <ArrowRight className="h-3 w-3 shrink-0 text-slate-700" />
                      <span>{fmtCost(call.cost_usd)}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
