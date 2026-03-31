import type { CampaignQualityDiagnostics } from '../types'

interface CampaignQualityDiagnosticsPanelProps {
  data?: CampaignQualityDiagnostics | null
  loading?: boolean
  title?: string
}

function SummaryCard({
  label,
  value,
  caption,
}: {
  label: string
  value: string
  caption?: string | null
}) {
  return (
    <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </p>
      <p className="mt-2 text-lg font-semibold text-white">{value}</p>
      {caption ? <p className="mt-1 text-xs text-slate-400">{caption}</p> : null}
    </div>
  )
}

function SectionList({
  title,
  items,
  emptyLabel,
  labelKey,
  valueKey,
}: {
  title: string
  items: Record<string, string | number>[]
  emptyLabel: string
  labelKey: string
  valueKey: string
}) {
  return (
    <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
      <h3 className="text-sm font-medium text-white">{title}</h3>
      {items.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">{emptyLabel}</p>
      ) : (
        <div className="mt-3 space-y-2">
          {items.map((item) => (
            <div
              key={`${title}-${String(item[labelKey])}`}
              className="flex items-start justify-between gap-3 text-sm"
            >
              <span className="break-words text-slate-300">
                {String(item[labelKey] ?? '')}
              </span>
              <span className="shrink-0 text-cyan-400">
                {String(item[valueKey] ?? 0)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function CampaignQualityDiagnosticsPanel({
  data,
  loading = false,
  title = 'Campaign Failure Diagnostics',
}: CampaignQualityDiagnosticsPanelProps) {
  if (loading) {
    return (
      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-6">
        <div className="animate-pulse space-y-3">
          <div className="h-4 w-48 rounded bg-slate-700/50" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            {[0, 1, 2, 3].map((item) => (
              <div key={item} className="h-24 rounded-xl bg-slate-800/60" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  const diagnostics = data ?? null
  const topCause = diagnostics?.by_cause_type?.[0] ?? null
  const topBlocker = diagnostics?.top_primary_blockers?.[0] ?? null
  const topMissingInput = diagnostics?.top_missing_inputs?.[0] ?? null
  const topVendor = diagnostics?.top_vendors?.[0] ?? null
  const hasAnyData =
    (diagnostics?.by_boundary?.length ?? 0) > 0 ||
    (diagnostics?.by_cause_type?.length ?? 0) > 0 ||
    (diagnostics?.top_primary_blockers?.length ?? 0) > 0 ||
    (diagnostics?.top_missing_inputs?.length ?? 0) > 0 ||
    (diagnostics?.by_target_mode?.length ?? 0) > 0 ||
    (diagnostics?.top_vendors?.length ?? 0) > 0

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="mt-1 text-sm text-slate-400">
            Why campaigns fail, where they fail, and whether evidence was missing or ignored.
          </p>
        </div>
        {diagnostics ? (
          <span className="rounded-full border border-slate-700/50 bg-slate-900/60 px-3 py-1 text-xs text-slate-300">
            last {diagnostics.days} days
          </span>
        ) : null}
      </div>

      {!hasAnyData ? (
        <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-6 text-sm text-slate-500">
          No failure diagnostics available yet.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <SummaryCard
              label="Top Cause"
              value={topCause?.cause_type ?? 'None'}
              caption={topCause ? `${topCause.count} failures` : null}
            />
            <SummaryCard
              label="Top Blocker"
              value={topBlocker?.reason ?? 'None'}
              caption={topBlocker ? `${topBlocker.count} hits` : null}
            />
            <SummaryCard
              label="Missing Input"
              value={topMissingInput?.input ?? 'None'}
              caption={topMissingInput ? `${topMissingInput.count} failures` : null}
            />
            <SummaryCard
              label="Top Vendor"
              value={topVendor?.vendor_name ?? 'None'}
              caption={topVendor ? `${topVendor.count} failures` : null}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <SectionList
              title="Failures By Boundary"
              items={diagnostics?.by_boundary ?? []}
              emptyLabel="No boundary failures yet."
              labelKey="boundary"
              valueKey="count"
            />
            <SectionList
              title="Failures By Cause Type"
              items={diagnostics?.by_cause_type ?? []}
              emptyLabel="No cause classifications yet."
              labelKey="cause_type"
              valueKey="count"
            />
            <SectionList
              title="Top Primary Blockers"
              items={diagnostics?.top_primary_blockers ?? []}
              emptyLabel="No blocker reasons yet."
              labelKey="reason"
              valueKey="count"
            />
            <SectionList
              title="Top Missing Inputs"
              items={diagnostics?.top_missing_inputs ?? []}
              emptyLabel="No missing input failures yet."
              labelKey="input"
              valueKey="count"
            />
            <SectionList
              title="Failures By Target Mode"
              items={diagnostics?.by_target_mode ?? []}
              emptyLabel="No target-mode failures yet."
              labelKey="target_mode"
              valueKey="count"
            />
            <SectionList
              title="Top Vendors"
              items={diagnostics?.top_vendors ?? []}
              emptyLabel="No vendor failures yet."
              labelKey="vendor_name"
              valueKey="count"
            />
          </div>
        </>
      )}
    </section>
  )
}
