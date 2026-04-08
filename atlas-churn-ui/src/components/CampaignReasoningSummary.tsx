import type {
  CampaignReasoningAtomContext,
  CampaignReasoningDeltaSummary,
  CampaignReasoningScopeSummary,
} from '../types'

interface CampaignReasoningSummaryModel {
  company_name: string
  reasoning_scope_summary?: CampaignReasoningScopeSummary
  reasoning_atom_context?: CampaignReasoningAtomContext
  reasoning_delta_summary?: CampaignReasoningDeltaSummary
}

export default function CampaignReasoningSummary({
  item,
}: {
  item: CampaignReasoningSummaryModel
}) {
  const scope = item.reasoning_scope_summary
  const atom = item.reasoning_atom_context
  const delta = item.reasoning_delta_summary
  const theses = atom?.top_theses ?? []
  const timingWindows = atom?.timing_windows ?? []
  const proofPoints = atom?.proof_points ?? []
  const accountSignals = atom?.account_signals ?? []
  const coverageLimits = atom?.coverage_limits ?? []

  const deltaSignals = [
    ...(delta?.theses_added ?? []).map((entry) => `new thesis: ${entry}`),
    ...(delta?.new_timing_windows ?? []).map((entry) => `new timing: ${entry}`),
    ...(delta?.new_account_signals ?? []).map((entry) => `new account: ${entry}`),
  ]
  if (delta?.wedge_changed) deltaSignals.unshift('wedge changed')
  if (delta?.confidence_changed) deltaSignals.push('confidence shifted')
  if (delta?.top_destination_changed) deltaSignals.push('destination competitor changed')

  if (
    !scope &&
    theses.length === 0 &&
    timingWindows.length === 0 &&
    proofPoints.length === 0 &&
    accountSignals.length === 0 &&
    coverageLimits.length === 0 &&
    deltaSignals.length === 0
  ) {
    return null
  }

  return (
    <div className="mb-3 rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        {scope?.selection_strategy && (
          <span className="rounded-full bg-slate-800/70 px-2 py-0.5 text-slate-300">
            {scope.selection_strategy}
          </span>
        )}
        {scope?.witnesses_in_scope != null && (
          <span className="rounded-full bg-slate-800/70 px-2 py-0.5 text-slate-300">
            {scope.witnesses_in_scope} witnesses
          </span>
        )}
        {(scope?.witness_mix ? Object.entries(scope.witness_mix) : []).slice(0, 3).map(([label, count]) => (
          <span key={label} className="rounded-full bg-slate-800/70 px-2 py-0.5 text-slate-400">
            {label}: {count}
          </span>
        ))}
        {coverageLimits.slice(0, 3).map((limit) => (
          <span key={limit} className="rounded-full bg-amber-500/15 px-2 py-0.5 text-amber-300">
            {limit}
          </span>
        ))}
      </div>

      {theses[0] && (
        <div className="mt-3 rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
          <p className="text-xs font-medium uppercase tracking-wide text-cyan-300">Top Thesis</p>
          <p className="mt-1 text-sm text-slate-100">{theses[0].summary}</p>
          {theses[0].why_now && (
            <p className="mt-1 text-xs text-slate-400">{theses[0].why_now}</p>
          )}
        </div>
      )}

      {(timingWindows.length > 0 || proofPoints.length > 0) && (
        <div className="mt-3 grid gap-3 lg:grid-cols-2">
          {timingWindows.length > 0 && (
            <div className="rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-300">Timing Windows</p>
              <div className="mt-2 space-y-2">
                {timingWindows.slice(0, 2).map((entry) => (
                  <div key={`${entry.window_type}:${entry.anchor}`} className="text-xs text-slate-300">
                    <p className="font-medium text-slate-200">{entry.anchor}</p>
                    <p className="text-slate-400">
                      {[entry.window_type, entry.urgency].filter(Boolean).join(' | ')}
                    </p>
                    {entry.recommended_action && (
                      <p className="mt-1 text-slate-400">{entry.recommended_action}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {proofPoints.length > 0 && (
            <div className="rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-300">Proof Points</p>
              <div className="mt-2 space-y-2">
                {proofPoints.slice(0, 2).map((entry) => (
                  <div key={entry.label} className="text-xs text-slate-300">
                    <p className="font-medium text-slate-200">{entry.label}</p>
                    {entry.value != null && (
                      <p className="text-slate-400">{String(entry.value)}</p>
                    )}
                    {entry.interpretation && (
                      <p className="mt-1 text-slate-400">{entry.interpretation}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {accountSignals.length > 0 && (
        <div className="mt-3 rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-300">Account Signals</p>
          <div className="mt-2 grid gap-2 lg:grid-cols-2">
            {accountSignals.slice(0, 2).map((signal, index) => (
              <div key={`${signal.company || 'account'}:${index}`} className="rounded-lg border border-slate-700/30 bg-slate-800/40 p-3 text-xs text-slate-300">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-medium text-slate-100">{signal.company || item.company_name}</p>
                  {signal.trust_tier && (
                    <span className="rounded-full bg-slate-700/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-300">
                      {signal.trust_tier}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-slate-400">
                  {[signal.role_type, signal.buying_stage].filter(Boolean).join(' | ')}
                </p>
                <p className="mt-1 text-slate-400">
                  {[signal.primary_pain, signal.competitor_context].filter(Boolean).join(' | ')}
                </p>
                <p className="mt-1 text-slate-400">
                  {[signal.contract_end, signal.decision_timeline].filter(Boolean).join(' | ')}
                </p>
                {signal.urgency != null && signal.urgency !== '' && (
                  <p className="mt-1 text-cyan-300">Urgency: {String(signal.urgency)}</p>
                )}
                {signal.quote && (
                  <p className="mt-2 line-clamp-3 text-slate-300">"{signal.quote}"</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {deltaSignals.length > 0 && (
        <div className="mt-3 rounded-lg border border-slate-700/40 bg-slate-900/40 p-3">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-300">Recent Change</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {deltaSignals.slice(0, 4).map((entry) => (
              <span key={entry} className="rounded-full bg-cyan-500/10 px-2 py-0.5 text-[11px] text-cyan-200">
                {entry}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
