import type { BlogQualityDiagnostics } from '../types'

interface BlogQualityDiagnosticsPanelProps {
  data?: BlogQualityDiagnostics | null
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

export default function BlogQualityDiagnosticsPanel({
  data,
  loading = false,
  title = 'Blog Failure Diagnostics',
}: BlogQualityDiagnosticsPanelProps) {
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
  const activeFailureCount = diagnostics?.active_failure_count ?? 0
  const rejectedFailureCount = diagnostics?.rejected_failure_count ?? 0
  const blockedSlugCount = diagnostics?.current_blocked_slug_count ?? 0
  const retryLimitBlockedCount = diagnostics?.retry_limit_blocked_slug_count ?? 0
  const topCause = diagnostics?.by_cause_type?.[0] ?? null
  const topBlocker = diagnostics?.top_primary_blockers?.[0] ?? null
  const hasAnyData =
    (diagnostics?.by_status?.length ?? 0) > 0 ||
    (diagnostics?.by_boundary?.length ?? 0) > 0 ||
    (diagnostics?.by_cause_type?.length ?? 0) > 0 ||
    (diagnostics?.top_primary_blockers?.length ?? 0) > 0 ||
    (diagnostics?.top_missing_inputs?.length ?? 0) > 0 ||
    (diagnostics?.by_topic_type?.length ?? 0) > 0 ||
    (diagnostics?.top_subjects?.length ?? 0) > 0 ||
    (diagnostics?.top_blocked_slugs?.length ?? 0) > 0

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="mt-1 text-sm text-slate-400">
            Why blog drafts fail, where they fail, and whether evidence or context was missing.
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
          No blog failure diagnostics available yet.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-6">
            <SummaryCard
              label="Active Draft Failures"
              value={String(activeFailureCount)}
              caption="Failing drafts still in the review queue"
            />
            <SummaryCard
              label="Rejected Failures"
              value={String(rejectedFailureCount)}
              caption="Intentionally retired drafts with failed audits"
            />
            <SummaryCard
              label="Top Cause (All Failed Rows)"
              value={topCause?.cause_type ?? 'None'}
              caption={
                topCause
                  ? `${topCause.count} failures${activeFailureCount === 0 && rejectedFailureCount > 0 ? ' (rejected only)' : ''}`
                  : null
              }
            />
            <SummaryCard
              label="Top Blocker (All Failed Rows)"
              value={topBlocker?.reason ?? 'None'}
              caption={
                topBlocker
                  ? `${topBlocker.count} hits${activeFailureCount === 0 && rejectedFailureCount > 0 ? ' (rejected only)' : ''}`
                  : null
              }
            />
            <SummaryCard
              label="Blocked Slugs"
              value={String(blockedSlugCount)}
              caption="Current rejected slugs blocked from regeneration"
            />
            <SummaryCard
              label="Retry Cap Hit"
              value={String(retryLimitBlockedCount)}
              caption="Blocked until manually force-regenerated"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <SectionList
              title="Failures By Status"
              items={diagnostics?.by_status ?? []}
              emptyLabel="No status-split failures yet."
              labelKey="status"
              valueKey="count"
            />
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
              emptyLabel="No missing-input failures yet."
              labelKey="input"
              valueKey="count"
            />
            <SectionList
              title="Failures By Topic Type"
              items={diagnostics?.by_topic_type ?? []}
              emptyLabel="No topic-type failures yet."
              labelKey="topic_type"
              valueKey="count"
            />
            <SectionList
              title="Top Subjects"
              items={diagnostics?.top_subjects ?? []}
              emptyLabel="No subject failures yet."
              labelKey="subject"
              valueKey="count"
            />
            <SectionList
              title="Blocked Slugs"
              items={diagnostics?.top_blocked_slugs ?? []}
              emptyLabel="No blocked blog slugs right now."
              labelKey="slug"
              valueKey="reason"
            />
          </div>
        </>
      )}
    </section>
  )
}
