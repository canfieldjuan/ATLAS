import type { CampaignFailureExplanation as CampaignFailureExplanationData } from '../types'

interface CampaignFailureExplanationProps {
  explanation?: CampaignFailureExplanationData | null
  boundaryLabel?: string | null
}

function formatCauseLabel(value: string): string {
  return value
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function DetailList({
  label,
  items,
  tone = 'text-slate-300',
}: {
  label: string
  items: string[]
  tone?: string
}) {
  if (items.length === 0) return null
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wider text-slate-500">{label}</p>
      <div className="mt-1 flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span
            key={`${label}-${item}`}
            className={`rounded-full border border-slate-700/50 bg-slate-900/60 px-2 py-1 text-[11px] ${tone}`}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  )
}

export default function CampaignFailureExplanation({
  explanation,
  boundaryLabel,
}: CampaignFailureExplanationProps) {
  if (!explanation) return null

  const blockers = explanation.blocking_issues ?? []
  const missingInputs = explanation.missing_inputs ?? []
  const missingPrimaryInputs = explanation.missing_primary_inputs ?? []
  const unusedProofTerms = explanation.unused_proof_terms ?? []
  const usedProofTerms = explanation.used_proof_terms ?? []
  const missingGroups = explanation.missing_groups ?? []
  const availableGroups = explanation.available_groups ?? []
  const hasAnyDetail =
    blockers.length > 0 ||
    missingInputs.length > 0 ||
    missingPrimaryInputs.length > 0 ||
    unusedProofTerms.length > 0 ||
    usedProofTerms.length > 0 ||
    missingGroups.length > 0

  if (!hasAnyDetail) return null

  const boundary = boundaryLabel ?? explanation.boundary ?? null
  const causeType = explanation.cause_type ? formatCauseLabel(explanation.cause_type) : null
  const primaryBlocker = explanation.primary_blocker ?? blockers[0] ?? null
  const sourceSummary = explanation.context_sources?.length
    ? explanation.context_sources.join(', ')
    : 'none'

  return (
    <section className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-red-300">
            Why This Failed
          </p>
          {primaryBlocker && (
            <p className="mt-1 text-sm text-red-100">{primaryBlocker}</p>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          {causeType && (
            <span className="rounded-full border border-red-500/25 bg-red-500/10 px-2 py-1 text-red-200">
              {causeType}
            </span>
          )}
          {boundary && (
            <span className="rounded-full border border-slate-700/50 bg-slate-900/60 px-2 py-1 text-slate-300">
              {boundary}
            </span>
          )}
        </div>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <DetailList label="Missing Inputs" items={missingInputs} tone="text-amber-200" />
        <DetailList label="Unavailable Before Fallback" items={missingPrimaryInputs} tone="text-amber-200" />
        <DetailList label="Available But Unused" items={unusedProofTerms} tone="text-red-200" />
        <DetailList label="Matched Proof Terms" items={usedProofTerms} tone="text-green-200" />
        <DetailList label="Missing Signal Groups" items={missingGroups} tone="text-red-200" />
        <DetailList label="Available Signal Groups" items={availableGroups} />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
        <span>sources: {sourceSummary}</span>
        <span>anchors: {explanation.anchor_count ?? 0}</span>
        <span>highlights: {explanation.highlight_count ?? 0}</span>
        <span>witness refs: {explanation.reference_id_counts?.witness_ids ?? 0}</span>
        {explanation.fallback_used && <span>reasoning fallback used</span>}
        {explanation.reasoning_view_found === false && <span>no reasoning view found</span>}
      </div>
    </section>
  )
}
