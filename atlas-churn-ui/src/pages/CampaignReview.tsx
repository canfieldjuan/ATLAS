import { useState, useMemo } from 'react'
import {
  MailSearch,
  RefreshCw,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  User,
  Clock,
  Loader2,
  AlertTriangle,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '../hooks/useApiData'
import StatCard from '../components/StatCard'
import type { ReviewQueueDraft, AuditEvent } from '../types'
import {
  fetchReviewQueue,
  fetchReviewQueueSummary,
  fetchCampaignAuditLog,
  bulkApproveCampaigns,
  bulkRejectCampaigns,
} from '../api/client'

type StatusTab = 'draft' | 'approved' | 'sent' | 'all'

interface CompanyGroup {
  company: string
  drafts: ReviewQueueDraft[]
}

function PersonaBadge({ persona }: { persona: string | null }) {
  if (!persona) return null
  const colors: Record<string, string> = {
    executive: 'bg-purple-500/20 text-purple-400',
    technical: 'bg-blue-500/20 text-blue-400',
    operations: 'bg-amber-500/20 text-amber-400',
  }
  const key = persona.toLowerCase()
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium', colors[key] || 'bg-slate-500/20 text-slate-400')}>
      {persona}
    </span>
  )
}

function CampaignStatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    draft: 'bg-amber-500/20 text-amber-400',
    approved: 'bg-green-500/20 text-green-400',
    queued: 'bg-cyan-500/20 text-cyan-400',
    sent: 'bg-cyan-500/20 text-cyan-400',
    cancelled: 'bg-red-500/20 text-red-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', styles[status] || 'bg-slate-500/20 text-slate-400')}>
      {status}
    </span>
  )
}

function AuditTimeline({ campaignId }: { campaignId: string }) {
  const { data, loading } = useApiData(
    () => fetchCampaignAuditLog(campaignId),
    [campaignId],
  )

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-slate-500 py-2">
        <Loader2 className="h-3 w-3 animate-spin" /> Loading audit log...
      </div>
    )
  }

  const events = data?.audit_log ?? []
  if (events.length === 0) {
    return <p className="text-xs text-slate-500 py-2">No audit events yet.</p>
  }

  return (
    <div className="space-y-1.5 py-2">
      {events.map((ev: AuditEvent) => (
        <div key={ev.id} className="flex items-center gap-2 text-xs">
          <Clock className="h-3 w-3 text-slate-500 shrink-0" />
          <span className="text-slate-400">
            {ev.created_at ? new Date(ev.created_at).toLocaleString() : '--'}
          </span>
          <CampaignStatusBadge status={ev.event_type} />
          {ev.source && <span className="text-slate-500">via {ev.source}</span>}
        </div>
      ))}
    </div>
  )
}

export default function CampaignReview() {
  const [statusTab, setStatusTab] = useState<StatusTab>('draft')
  const [expandedCompany, setExpandedCompany] = useState<string | null>(null)
  const [selectedCampaign, setSelectedCampaign] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [bulkLoading, setBulkLoading] = useState(false)

  const { data: summaryData, loading: summaryLoading } = useApiData(
    fetchReviewQueueSummary,
    [],
  )

  const { data: queueData, loading, error, refresh, refreshing } = useApiData(
    () => fetchReviewQueue({ status: statusTab, include_prospects: true, limit: 200 }),
    [statusTab],
  )

  const groups = useMemo<CompanyGroup[]>(() => {
    const drafts = queueData?.drafts ?? []
    const map = new Map<string, ReviewQueueDraft[]>()
    for (const d of drafts) {
      const key = d.company_name || 'Unknown'
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push(d)
    }
    return Array.from(map.entries())
      .map(([company, items]) => ({ company, drafts: items }))
      .sort((a, b) => b.drafts.length - a.drafts.length)
  }, [queueData])

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleBulkAction = async (action: 'approve' | 'queue-send' | 'reject') => {
    if (selectedIds.size === 0) return
    setBulkLoading(true)
    try {
      if (action === 'reject') {
        await bulkRejectCampaigns(Array.from(selectedIds))
      } else {
        await bulkApproveCampaigns(Array.from(selectedIds), action)
      }
      setSelectedIds(new Set())
      refresh()
    } finally {
      setBulkLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <MailSearch className="h-6 w-6 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Campaign Review</h1>
        </div>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Pending Review"
          value={summaryData?.pending_review ?? 0}
          icon={<MailSearch className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Ready to Send"
          value={summaryData?.ready_to_send ?? 0}
          icon={<CheckCircle2 className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Missing Recipient"
          value={summaryData?.pending_recipient ?? 0}
          icon={<User className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Suppressed"
          value={summaryData?.suppressed ?? 0}
          icon={<AlertTriangle className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
      </div>

      {/* Status tabs */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 border-b border-slate-700/50">
          {(['draft', 'approved', 'sent', 'all'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => { setStatusTab(tab); setExpandedCompany(null); setSelectedIds(new Set()) }}
              className={clsx(
                'px-4 py-2 text-sm font-medium transition-colors border-b-2 capitalize',
                statusTab === tab
                  ? 'text-cyan-400 border-cyan-400'
                  : 'text-slate-400 border-transparent hover:text-white',
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Bulk actions */}
        {selectedIds.size > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">{selectedIds.size} selected</span>
            <button
              onClick={() => handleBulkAction('approve')}
              disabled={bulkLoading}
              className="px-3 py-1.5 text-xs font-medium bg-green-600 hover:bg-green-500 text-white rounded-lg disabled:opacity-50"
            >
              Approve
            </button>
            <button
              onClick={() => handleBulkAction('reject')}
              disabled={bulkLoading}
              className="px-3 py-1.5 text-xs font-medium bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg disabled:opacity-50"
            >
              Reject
            </button>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 text-sm">
          {error.message}
        </div>
      )}

      {/* Company groups */}
      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 animate-pulse">
              <div className="h-5 w-40 bg-slate-700/50 rounded" />
            </div>
          ))}
        </div>
      ) : groups.length === 0 ? (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-12 text-center">
          <MailSearch className="h-10 w-10 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No campaigns match this filter.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {groups.map((group) => {
            const isExpanded = expandedCompany === group.company
            return (
              <div
                key={group.company}
                className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden"
              >
                {/* Company header */}
                <button
                  onClick={() => setExpandedCompany(isExpanded ? null : group.company)}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-800/30 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-slate-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-slate-400" />
                    )}
                    <span className="text-white font-medium">{group.company}</span>
                    <span className="text-xs text-slate-500">{group.drafts.length} campaigns</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Show unique personas */}
                    {[...new Set(group.drafts.map((d) => d.target_persona).filter(Boolean))].map((p) => (
                      <PersonaBadge key={p} persona={p!} />
                    ))}
                  </div>
                </button>

                {/* Expanded: campaign cards */}
                {isExpanded && (
                  <div className="border-t border-slate-700/50 divide-y divide-slate-700/30">
                    {group.drafts.map((draft) => (
                      <div key={draft.id} className="p-4">
                        <div className="flex items-start gap-3">
                          {/* Checkbox */}
                          {draft.status === 'draft' && (
                            <input
                              type="checkbox"
                              checked={selectedIds.has(draft.id)}
                              onChange={() => toggleSelect(draft.id)}
                              className="mt-1 accent-cyan-500"
                            />
                          )}

                          <div className="flex-1 min-w-0">
                            {/* Row 1: persona + prospect + status */}
                            <div className="flex items-center gap-2 mb-2 flex-wrap">
                              <PersonaBadge persona={draft.target_persona} />
                              <CampaignStatusBadge status={draft.status} />
                              {draft.is_suppressed > 0 && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
                                  suppressed
                                </span>
                              )}
                              {draft.step_number != null && (
                                <span className="text-xs text-slate-500">
                                  Step {draft.step_number}{draft.max_steps ? `/${draft.max_steps}` : ''}
                                </span>
                              )}
                            </div>

                            {/* Prospect info */}
                            {(draft.prospect_first_name || draft.recipient_email || draft.seq_recipient) && (
                              <div className="flex items-center gap-2 mb-2 text-xs">
                                <User className="h-3 w-3 text-slate-500" />
                                <span className="text-slate-300">
                                  {[draft.prospect_first_name, draft.prospect_last_name].filter(Boolean).join(' ') || 'Unknown'}
                                </span>
                                {draft.prospect_title && (
                                  <span className="text-slate-500">{draft.prospect_title}</span>
                                )}
                                {draft.prospect_seniority && (
                                  <span className="px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400">
                                    {draft.prospect_seniority}
                                  </span>
                                )}
                                <span className="text-slate-500">
                                  {draft.recipient_email || draft.seq_recipient || '--'}
                                </span>
                              </div>
                            )}

                            {/* Email preview */}
                            <div className="bg-slate-800/50 border border-slate-700/30 rounded-lg p-3 mb-2">
                              <p className="text-sm font-medium text-white mb-1">
                                {draft.subject || '(no subject)'}
                              </p>
                              {draft.body && (
                                <div
                                  className="text-xs text-slate-300 line-clamp-4 prose prose-invert prose-xs max-w-none"
                                  dangerouslySetInnerHTML={{ __html: draft.body }}
                                />
                              )}
                              {draft.cta && (
                                <p className="text-xs text-cyan-400 mt-2">{draft.cta}</p>
                              )}
                            </div>

                            {/* Audit timeline toggle */}
                            <button
                              onClick={() =>
                                setSelectedCampaign(selectedCampaign === draft.id ? null : draft.id)
                              }
                              className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
                            >
                              <Clock className="h-3 w-3" />
                              {selectedCampaign === draft.id ? 'Hide' : 'Show'} audit log
                            </button>
                            {selectedCampaign === draft.id && (
                              <AuditTimeline campaignId={draft.id} />
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
