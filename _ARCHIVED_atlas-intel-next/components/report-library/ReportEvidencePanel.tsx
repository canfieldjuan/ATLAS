import { AlertTriangle, FileSearch, Quote, ShieldCheck } from 'lucide-react'
import type { ReportDetail } from '@/lib/types'
import { extractReportEvidence, qualityStatusLabel } from '@/lib/reportLibrary'

function humanLabel(value: string): string {
  return value.replace(/_/g, ' ')
}

function qualityPillClass(status: string | undefined): string {
  switch (status) {
    case 'sales_ready':
      return 'bg-emerald-500/10 text-emerald-300'
    case 'needs_review':
      return 'bg-amber-500/10 text-amber-300'
    case 'thin_evidence':
      return 'bg-slate-800 text-slate-300'
    case 'deterministic_fallback':
      return 'bg-rose-500/10 text-rose-300'
    default:
      return 'bg-slate-800 text-slate-300'
  }
}

export default function ReportEvidencePanel({ report }: { report: ReportDetail }) {
  const evidence = extractReportEvidence(report.intelligence_data, report.data_density)
  const referenceCount = evidence.referenceIds.metricIds.length + evidence.referenceIds.witnessIds.length
  const hasOperatorNotes = evidence.qualityFailedChecks.length > 0 || evidence.qualityWarnings.length > 0
  const hasEvidence = evidence.witnesses.length > 0 || evidence.quotes.length > 0 || referenceCount > 0

  return (
    <section className="rounded-3xl border border-slate-700/60 bg-slate-900/60 p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-cyan-300">
            <FileSearch className="h-3.5 w-3.5" />
            Citation Panel
          </div>
          <h2 className="mt-4 text-lg font-semibold text-white">Witness-backed evidence</h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">
            Operator review status, reasoning provenance, and the strongest supporting witness snippets for this deliverable.
          </p>
        </div>
        <div className="rounded-2xl border border-slate-700/60 bg-slate-950/40 px-3 py-2 text-right">
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">References</p>
          <p className="mt-1 text-sm font-medium text-white">{referenceCount}</p>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {evidence.qualityStatus && (
          <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${qualityPillClass(evidence.qualityStatus)}`}>
            <ShieldCheck className="h-3.5 w-3.5" />
            {qualityStatusLabel(evidence.qualityStatus)}
            {typeof evidence.qualityScore === 'number' ? ` Score ${Math.round(evidence.qualityScore)}` : ''}
          </span>
        )}
        {evidence.llmRenderStatus && (
          <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
            Render: {humanLabel(evidence.llmRenderStatus)}
          </span>
        )}
        {evidence.dataDensityStatus && (
          <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
            Data density: {humanLabel(evidence.dataDensityStatus)}
          </span>
        )}
        {evidence.reasoningSources.map((source) => (
          <span key={source} className="rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
            Source: {humanLabel(source)}
          </span>
        ))}
        {evidence.referenceIds.witnessIds.length > 0 && (
          <span className="rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-300">
            {evidence.referenceIds.witnessIds.length} witness refs
          </span>
        )}
        {evidence.referenceIds.metricIds.length > 0 && (
          <span className="rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-300">
            {evidence.referenceIds.metricIds.length} metric refs
          </span>
        )}
      </div>

      {!hasEvidence && !hasOperatorNotes ? (
        <div className="mt-5 rounded-2xl border border-dashed border-slate-700/60 bg-slate-950/30 p-4 text-sm text-slate-400">
          This artifact does not expose witness highlights yet. The detail surface is ready for evidence drawers as soon as those fields are attached.
        </div>
      ) : null}

      {hasOperatorNotes && (
        <div className="mt-5 rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4">
          <div className="flex items-center gap-2 text-sm font-medium text-amber-300">
            <AlertTriangle className="h-4 w-4" />
            Operator review status
          </div>
          {evidence.qualityFailedChecks.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm text-amber-100/90">
              {evidence.qualityFailedChecks.map((item) => (
                <li key={item} className="flex gap-2">
                  <span className="text-amber-400">-</span>
                  <span>{humanLabel(item)}</span>
                </li>
              ))}
            </ul>
          )}
          {evidence.qualityWarnings.length > 0 && (
            <ul className="mt-3 space-y-1 text-sm text-slate-300">
              {evidence.qualityWarnings.map((item) => (
                <li key={item} className="flex gap-2">
                  <span className="text-slate-500">-</span>
                  <span>{humanLabel(item)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {evidence.witnesses.length > 0 && (
        <div className="mt-5 space-y-3">
          <h3 className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Top witnesses</h3>
          {evidence.witnesses.slice(0, 4).map((witness) => (
            <article key={witness.key} className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
              <div className="flex flex-wrap items-center gap-2">
                {witness.label && (
                  <span className="rounded-full bg-cyan-500/10 px-2.5 py-1 text-[11px] font-medium text-cyan-300">
                    {humanLabel(witness.label)}
                  </span>
                )}
                {witness.salienceScore != null && (
                  <span className="rounded-full bg-slate-800 px-2.5 py-1 text-[11px] font-medium text-slate-300">
                    Salience {witness.salienceScore.toFixed(1)}
                  </span>
                )}
                {witness.id && (
                  <span className="rounded-full bg-slate-800 px-2.5 py-1 text-[11px] font-medium text-slate-300">
                    {witness.id}
                  </span>
                )}
              </div>
              {witness.excerptText && (
                <blockquote className="mt-3 border-l-2 border-cyan-500/40 pl-3 text-sm leading-6 text-slate-200">
                  &quot;{witness.excerptText}&quot;
                </blockquote>
              )}
              <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                {witness.reviewerCompany && <span>{witness.reviewerCompany}</span>}
                {witness.reviewerTitle && <span>{witness.reviewerTitle}</span>}
                {witness.timeAnchor && <span>{witness.timeAnchor}</span>}
                {witness.competitor && <span>vs {witness.competitor}</span>}
                {witness.witnessType && <span>{humanLabel(witness.witnessType)}</span>}
              </div>
              {witness.selectionReason && (
                <p className="mt-2 text-xs text-slate-400">{witness.selectionReason}</p>
              )}
              {witness.numericTokens.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {witness.numericTokens.map((token) => (
                    <span key={token} className="rounded-full bg-slate-800 px-2.5 py-1 text-[11px] font-medium text-slate-300">
                      {token}
                    </span>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>
      )}

      {evidence.quotes.length > 0 && (
        <div className="mt-5 space-y-3">
          <h3 className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Quoted evidence</h3>
          {evidence.quotes.slice(0, 3).map((quote) => (
            <article key={quote.key} className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
              <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
                <Quote className="h-3.5 w-3.5" />
                Quote
              </div>
              <blockquote className="mt-3 border-l-2 border-amber-500/40 pl-3 text-sm leading-6 text-slate-200">
                &quot;{quote.quote}&quot;
              </blockquote>
              <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                {quote.company && <span>{quote.company}</span>}
                {quote.role && <span>{quote.role}</span>}
                {quote.sourceSite && <span>{quote.sourceSite}</span>}
                {quote.painCategory && <span>{humanLabel(quote.painCategory)}</span>}
                {quote.urgency != null && <span>Urgency {quote.urgency}/10</span>}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
