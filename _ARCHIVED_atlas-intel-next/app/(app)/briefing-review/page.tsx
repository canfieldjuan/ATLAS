"use client";

import { useState, useMemo } from 'react'
import {
  Mail,
  RefreshCw,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronRight,
  Clock,
  Send,
  Eye,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '@/lib/hooks/useApiData'
import StatCard from '@/components/StatCard'
import type { BriefingDraft } from '@/lib/types'
import {
  fetchBriefingReviewQueue,
  fetchBriefingReviewSummary,
  bulkApproveBriefings,
  bulkRejectBriefings,
} from '@/lib/api/client'

type StatusTab = 'pending_approval' | 'sent' | 'rejected' | 'all'

interface VendorGroup {
  vendor: string
  briefings: BriefingDraft[]
}

function BriefingStatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending_approval: 'bg-amber-500/20 text-amber-400',
    sent: 'bg-green-500/20 text-green-400',
    opened: 'bg-cyan-500/20 text-cyan-400',
    clicked: 'bg-cyan-500/20 text-cyan-400',
    rejected: 'bg-red-500/20 text-red-400',
    failed: 'bg-red-500/20 text-red-400',
    suppressed: 'bg-slate-500/20 text-slate-400',
  }
  const label = status === 'pending_approval' ? 'pending' : status
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', styles[status] || 'bg-slate-500/20 text-slate-400')}>
      {label}
    </span>
  )
}

function TargetModeBadge({ mode }: { mode: string | null }) {
  if (!mode) return null
  const isChallenger = mode === 'challenger_intel'
  return (
    <span className={clsx(
      'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
      isChallenger ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400',
    )}>
      {isChallenger ? 'challenger' : 'retention'}
    </span>
  )
}

function HtmlPreviewModal({ html, subject, onClose }: { html: string; subject: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h3 className="text-gray-900 font-medium text-sm truncate">{subject}</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-1">
          <iframe
            srcDoc={html}
            className="w-full h-full min-h-[60vh] border-0"
            title="Briefing preview"
            sandbox="allow-same-origin"
          />
        </div>
      </div>
    </div>
  )
}

export default function BriefingReview() {
  const [statusTab, setStatusTab] = useState<StatusTab>('pending_approval')
  const [expandedVendor, setExpandedVendor] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [bulkLoading, setBulkLoading] = useState(false)
  const [previewBriefing, setPreviewBriefing] = useState<BriefingDraft | null>(null)

  const { data: summaryData, loading: summaryLoading, refresh: refreshSummary } = useApiData(
    fetchBriefingReviewSummary,
    [],
  )

  const { data: queueData, loading, error, refresh, refreshing } = useApiData(
    () => fetchBriefingReviewQueue({ status: statusTab, limit: 200 }),
    [statusTab],
  )

  const groups = useMemo<VendorGroup[]>(() => {
    const briefings = queueData?.briefings ?? []
    const map = new Map<string, BriefingDraft[]>()
    for (const b of briefings) {
      const key = b.vendor_name || 'Unknown'
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push(b)
    }
    return Array.from(map.entries())
      .map(([vendor, items]) => ({ vendor, briefings: items }))
      .sort((a, b) => b.briefings.length - a.briefings.length)
  }, [queueData])

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleBulkAction = async (action: 'approve' | 'reject') => {
    if (selectedIds.size === 0) return
    setBulkLoading(true)
    try {
      if (action === 'reject') {
        await bulkRejectBriefings(Array.from(selectedIds))
      } else {
        await bulkApproveBriefings(Array.from(selectedIds))
      }
      setSelectedIds(new Set())
      refresh()
      refreshSummary()
    } finally {
      setBulkLoading(false)
    }
  }

  const refreshAll = () => {
    refresh()
    refreshSummary()
  }

  return (
    <div className="space-y-6">
      {/* Preview modal */}
      {previewBriefing?.briefing_html && (
        <HtmlPreviewModal
          html={previewBriefing.briefing_html}
          subject={previewBriefing.subject || previewBriefing.vendor_name}
          onClose={() => setPreviewBriefing(null)}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Mail className="h-6 w-6 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Briefing Review</h1>
        </div>
        <button
          onClick={refreshAll}
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
          label="Pending Approval"
          value={summaryData?.pending_approval ?? 0}
          icon={<Clock className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Sent"
          value={summaryData?.sent ?? 0}
          icon={<Send className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Rejected"
          value={summaryData?.rejected ?? 0}
          icon={<XCircle className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Failed"
          value={summaryData?.failed ?? 0}
          icon={<XCircle className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
      </div>

      {/* Oldest pending alert */}
      {summaryData?.oldest_pending_hours != null && summaryData.oldest_pending_hours > 24 && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 text-amber-400 text-sm">
          Oldest pending briefing: {Math.round(summaryData.oldest_pending_hours)}h ago
        </div>
      )}

      {/* Status tabs + bulk actions */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 border-b border-slate-700/50">
          {(['pending_approval', 'sent', 'rejected', 'all'] as const).map((tab) => {
            const labels: Record<StatusTab, string> = {
              pending_approval: 'Pending',
              sent: 'Sent',
              rejected: 'Rejected',
              all: 'All',
            }
            return (
              <button
                key={tab}
                onClick={() => { setStatusTab(tab); setExpandedVendor(null); setSelectedIds(new Set()) }}
                className={clsx(
                  'px-4 py-2 text-sm font-medium transition-colors border-b-2',
                  statusTab === tab
                    ? 'text-cyan-400 border-cyan-400'
                    : 'text-slate-400 border-transparent hover:text-white',
                )}
              >
                {labels[tab]}
              </button>
            )
          })}
        </div>

        {selectedIds.size > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">{selectedIds.size} selected</span>
            <button
              onClick={() => handleBulkAction('approve')}
              disabled={bulkLoading}
              className="px-3 py-1.5 text-xs font-medium bg-green-600 hover:bg-green-500 text-white rounded-lg disabled:opacity-50"
            >
              <CheckCircle2 className="h-3 w-3 inline mr-1" />
              Approve & Send
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

      {/* Vendor groups */}
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
          <Mail className="h-10 w-10 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No briefings match this filter.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {groups.map((group) => {
            const isExpanded = expandedVendor === group.vendor
            const pendingCount = group.briefings.filter(b => b.status === 'pending_approval').length
            return (
              <div
                key={group.vendor}
                className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden"
              >
                {/* Vendor header */}
                <button
                  onClick={() => setExpandedVendor(isExpanded ? null : group.vendor)}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-800/30 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-slate-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-slate-400" />
                    )}
                    <span className="text-white font-medium">{group.vendor}</span>
                    <span className="text-xs text-slate-500">
                      {group.briefings.length} briefing{group.briefings.length !== 1 ? 's' : ''}
                    </span>
                    {pendingCount > 0 && (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
                        {pendingCount} pending
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {[...new Set(group.briefings.map(b => b.target_mode).filter(Boolean))].map(m => (
                      <TargetModeBadge key={m} mode={m} />
                    ))}
                  </div>
                </button>

                {/* Expanded: briefing cards */}
                {isExpanded && (
                  <div className="border-t border-slate-700/50 divide-y divide-slate-700/30">
                    {group.briefings.map((briefing) => (
                      <div key={briefing.id} className="p-4">
                        <div className="flex items-start gap-3">
                          {/* Checkbox for pending */}
                          {briefing.status === 'pending_approval' && (
                            <input
                              type="checkbox"
                              checked={selectedIds.has(briefing.id)}
                              onChange={() => toggleSelect(briefing.id)}
                              className="mt-1 accent-cyan-500"
                            />
                          )}

                          <div className="flex-1 min-w-0">
                            {/* Row 1: badges */}
                            <div className="flex items-center gap-2 mb-2 flex-wrap">
                              <BriefingStatusBadge status={briefing.status} />
                              <TargetModeBadge mode={briefing.target_mode} />
                              {briefing.rejected_at && briefing.reject_reason && (
                                <span className="text-xs text-red-400">
                                  Reason: {briefing.reject_reason}
                                </span>
                              )}
                            </div>

                            {/* Recipient */}
                            <div className="flex items-center gap-2 mb-2 text-xs text-slate-400">
                              <Mail className="h-3 w-3" />
                              <span className="text-slate-300">{briefing.recipient_email}</span>
                              {briefing.created_at && (
                                <>
                                  <Clock className="h-3 w-3 ml-2" />
                                  <span>{new Date(briefing.created_at).toLocaleString()}</span>
                                </>
                              )}
                            </div>

                            {/* Subject + inline preview */}
                            <div className="bg-slate-800/50 border border-slate-700/30 rounded-lg p-3 mb-2">
                              <p className="text-sm font-medium text-white mb-1">
                                {briefing.subject || '(no subject)'}
                              </p>
                              {briefing.briefing_html && (
                                <p className="text-xs text-slate-400 mt-1">
                                  {briefing.briefing_html.replace(/<[^>]*>/g, '').slice(0, 200).trim()}...
                                </p>
                              )}
                            </div>

                            {/* Actions */}
                            {briefing.briefing_html && (
                              <button
                                onClick={() => setPreviewBriefing(briefing)}
                                className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                              >
                                <Eye className="h-3 w-3" />
                                Full preview
                              </button>
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
