"use client";

import { useState, useEffect } from 'react'
import {
  Users,
  RefreshCw,
  Search,
  UserCheck,
  UserPlus,
  Mail,
  Loader2,
  Download,
  X,
  Plus,
  Pencil,
  Trash2,
  RotateCcw,
  XCircle,
  Building2,
  ExternalLink,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '@/lib/hooks/useApiData'
import DataTable from '@/components/DataTable'
import StatCard from '@/components/StatCard'
import type { Column } from '@/components/DataTable'
import type { Prospect, ManualQueueEntry, CompanyOverride } from '@/lib/types'
import {
  fetchProspects,
  fetchProspectStats,
  fetchManualQueue,
  resolveManualQueueEntry,
  fetchCompanyOverrides,
  upsertCompanyOverride,
  deleteCompanyOverride,
  bootstrapCompanyOverrides,
  downloadProspectsCsv,
} from '@/lib/api/client'

// ---------------------------------------------------------------------------
// Badge helpers
// ---------------------------------------------------------------------------

function ProspectStatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    active: 'bg-green-500/20 text-green-400',
    contacted: 'bg-cyan-500/20 text-cyan-400',
    opted_out: 'bg-slate-500/20 text-slate-400',
    bounced: 'bg-red-500/20 text-red-400',
    suppressed: 'bg-amber-500/20 text-amber-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', styles[status] || 'bg-slate-500/20 text-slate-400')}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}

function EmailStatusBadge({ status }: { status: string | null }) {
  if (!status) return <span className="text-xs text-slate-500">--</span>
  const styles: Record<string, string> = {
    valid: 'text-green-400',
    invalid: 'text-red-400',
    catch_all: 'text-amber-400',
    unknown: 'text-slate-400',
  }
  return (
    <span className={clsx('text-xs', styles[status] || 'text-slate-400')}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}

function SeniorityBadge({ seniority }: { seniority: string | null }) {
  if (!seniority) return <span className="text-xs text-slate-500">--</span>
  const styles: Record<string, string> = {
    c_suite: 'bg-purple-500/20 text-purple-400',
    vp: 'bg-blue-500/20 text-blue-400',
    director: 'bg-cyan-500/20 text-cyan-400',
    manager: 'bg-green-500/20 text-green-400',
    senior: 'bg-amber-500/20 text-amber-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium', styles[seniority] || 'bg-slate-500/20 text-slate-400')}>
      {seniority.replace(/_/g, ' ')}
    </span>
  )
}

function SequenceStatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    active: 'bg-green-500/20 text-green-400',
    paused: 'bg-amber-500/20 text-amber-400',
    completed: 'bg-cyan-500/20 text-cyan-400',
    replied: 'bg-purple-500/20 text-purple-400',
    bounced: 'bg-red-500/20 text-red-400',
    unsubscribed: 'bg-slate-500/20 text-slate-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', styles[status] || 'bg-slate-500/20 text-slate-400')}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}

function QueueStatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    manual_review: 'bg-amber-500/20 text-amber-400',
    pending: 'bg-cyan-500/20 text-cyan-400',
    enriched: 'bg-green-500/20 text-green-400',
    not_found: 'bg-slate-500/20 text-slate-400',
    error: 'bg-red-500/20 text-red-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', styles[status] || 'bg-slate-500/20 text-slate-400')}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Error detail helpers
// ---------------------------------------------------------------------------

const ERROR_REASON_MAP: Record<string, { label: string; color: string }> = {
  apollo_no_results: { label: 'No Results', color: 'bg-slate-500/20 text-slate-400' },
  no_people_found: { label: 'No People', color: 'bg-amber-500/20 text-amber-400' },
  no_verified_email: { label: 'No Emails', color: 'bg-orange-500/20 text-orange-400' },
  manual_queue_seed_after_apollo_exhausted: { label: 'Apollo Exhausted', color: 'bg-red-500/20 text-red-400' },
}

function ErrorDetailCell({ detail }: { detail: string | null }) {
  if (!detail) return <span className="text-xs text-slate-500">--</span>

  let parsed: { reason?: string; search_names?: string[] } | null = null
  try {
    const raw = JSON.parse(detail)
    if (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) {
      parsed = raw
    }
  } catch {
    // not JSON — fall through to raw display
  }

  if (parsed?.reason) {
    const mapping = ERROR_REASON_MAP[parsed.reason]
    return (
      <div title={detail}>
        <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
          mapping?.color ?? 'bg-slate-500/20 text-slate-400')}>
          {mapping?.label ?? parsed.reason.slice(0, 30)}
        </span>
        {parsed.search_names && parsed.search_names.length > 0 && (
          <span className="block text-xs text-slate-500 mt-0.5 line-clamp-1 max-w-[180px]">
            {parsed.search_names.join(', ')}
          </span>
        )}
      </div>
    )
  }

  return (
    <span className="text-xs text-red-400 line-clamp-1 max-w-[200px]" title={detail}>
      {detail.slice(0, 60)}{detail.length > 60 ? '...' : ''}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Tab type
// ---------------------------------------------------------------------------

type ProspectsTab = 'prospects' | 'manual_queue' | 'company_overrides'

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prospect Detail Drawer
// ---------------------------------------------------------------------------

function ProspectDetailDrawer({
  prospect,
  onClose,
}: {
  prospect: Prospect
  onClose: () => void
}) {
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const fullName = [prospect.first_name, prospect.last_name].filter(Boolean).join(' ') || 'Unknown'
  const location = [prospect.city, prospect.state].filter(Boolean).join(', ')
  const hasReasoning =
    prospect.reasoning_scope_summary ||
    prospect.reasoning_atom_context ||
    prospect.reasoning_delta_summary

  return (
    <div
      className="fixed top-0 right-0 bottom-0 left-0 lg:left-56 flex items-center justify-center p-8"
      style={{ zIndex: 50, backgroundColor: 'rgba(0,0,0,0.7)' }}
      onClick={onClose}
    >
      <div
        className="bg-slate-800 border-2 border-slate-500 rounded-xl flex flex-col"
        style={{ maxWidth: '640px', width: '100%', maxHeight: '85vh', boxShadow: '0 0 40px rgba(0,0,0,0.8)' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 pb-3 border-b border-slate-700/50 shrink-0">
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-white truncate">{fullName}</h2>
            <div className="flex items-center gap-2 mt-1">
              <ProspectStatusBadge status={prospect.status} />
              {prospect.title && (
                <span className="text-xs text-slate-400 truncate">{prospect.title}</span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="ml-4 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="overflow-y-auto p-5 space-y-5 flex-1">
          {/* Contact */}
          <div>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Contact</span>
            <div className="mt-2 space-y-1.5">
              <div className="flex items-center gap-2">
                <Mail className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                {prospect.email ? (
                  <a href={`mailto:${prospect.email}`} className="text-sm text-cyan-400 hover:underline truncate">
                    {prospect.email}
                  </a>
                ) : (
                  <span className="text-sm text-slate-500">--</span>
                )}
                {prospect.email_status && <EmailStatusBadge status={prospect.email_status} />}
              </div>
              {prospect.linkedin_url && (
                <div className="flex items-center gap-2">
                  <ExternalLink className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                  <a
                    href={prospect.linkedin_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-cyan-400 hover:underline"
                  >
                    LinkedIn Profile
                  </a>
                </div>
              )}
              {prospect.seniority && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Seniority:</span>
                  <SeniorityBadge seniority={prospect.seniority} />
                </div>
              )}
            </div>
          </div>

          {/* Company */}
          <div>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Company</span>
            <div className="mt-2 space-y-1.5">
              <div className="flex items-center gap-2">
                <Building2 className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                <span className="text-sm text-white">{prospect.company_name || '--'}</span>
              </div>
              {prospect.company_domain && (
                <div className="flex items-center gap-2">
                  <ExternalLink className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                  <a
                    href={prospect.company_domain.startsWith("http") ? prospect.company_domain : `https://${prospect.company_domain}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-cyan-400 hover:underline"
                  >
                    {prospect.company_domain}
                  </a>
                </div>
              )}
              {location && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Location:</span>
                  <span className="text-sm text-slate-300">{location}</span>
                </div>
              )}
            </div>
          </div>

          {/* Sequence */}
          {prospect.related_sequence_status && (
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wider">Sequence</span>
              <div className="mt-2 space-y-1.5">
                <div className="flex items-center gap-2">
                  <SequenceStatusBadge status={prospect.related_sequence_status} />
                  {prospect.related_sequence_current_step != null && prospect.related_sequence_max_steps != null && (
                    <span className="text-sm text-slate-300">
                      Step {prospect.related_sequence_current_step}/{prospect.related_sequence_max_steps}
                    </span>
                  )}
                </div>
                {prospect.related_sequence_last_sent_at && (
                  <div className="text-xs text-slate-400">
                    Last sent: {new Date(prospect.related_sequence_last_sent_at).toLocaleDateString()}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Churning From */}
          {prospect.churning_from && (
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wider">Churning From</span>
              <div className="mt-2">
                <span className="text-sm text-amber-400">{prospect.churning_from}</span>
              </div>
            </div>
          )}

          {/* Reasoning Context */}
          {hasReasoning && (
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wider">Reasoning Context</span>
              <div className="mt-2 space-y-2">
                {prospect.reasoning_scope_summary && (
                  <div>
                    <span className="text-xs text-slate-500">Scope Summary</span>
                    <pre className="mt-1 text-xs text-slate-300 bg-slate-900/50 rounded-lg p-3 overflow-x-auto max-h-40">
                      {JSON.stringify(prospect.reasoning_scope_summary, null, 2)}
                    </pre>
                  </div>
                )}
                {prospect.reasoning_atom_context && (
                  <div>
                    <span className="text-xs text-slate-500">Atom Context</span>
                    <pre className="mt-1 text-xs text-slate-300 bg-slate-900/50 rounded-lg p-3 overflow-x-auto max-h-40">
                      {JSON.stringify(prospect.reasoning_atom_context, null, 2)}
                    </pre>
                  </div>
                )}
                {prospect.reasoning_delta_summary && (
                  <div>
                    <span className="text-xs text-slate-500">Delta Summary</span>
                    <pre className="mt-1 text-xs text-slate-300 bg-slate-900/50 rounded-lg p-3 overflow-x-auto max-h-40">
                      {JSON.stringify(prospect.reasoning_delta_summary, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ProspectsPage() {
  const [activeTab, setActiveTab] = useState<ProspectsTab>('prospects')
  const [actionResult, setActionResult] = useState<string | null>(null)
  const [isActionError, setIsActionError] = useState(false)
  const [viewingProspect, setViewingProspect] = useState<Prospect | null>(null)

  // ---- Prospects tab state ----
  const [companySearch, setCompanySearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [seniorityFilter, setSeniorityFilter] = useState('')

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(companySearch), 300)
    return () => clearTimeout(t)
  }, [companySearch])

  // ---- Manual Queue tab state ----
  const [mqSearch, setMqSearch] = useState('')
  const [mqDebouncedSearch, setMqDebouncedSearch] = useState('')
  const [resolvingId, setResolvingId] = useState<string | null>(null)
  const [resolveDomain, setResolveDomain] = useState('')
  const [resolveLoading, setResolveLoading] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setMqDebouncedSearch(mqSearch), 300)
    return () => clearTimeout(t)
  }, [mqSearch])

  // ---- Company Overrides tab state ----
  const [coSearch, setCoSearch] = useState('')
  const [coDebouncedSearch, setCoDebouncedSearch] = useState('')
  const [showOverrideForm, setShowOverrideForm] = useState(false)
  const [editingOverrideId, setEditingOverrideId] = useState<string | null>(null)
  const [overrideForm, setOverrideForm] = useState({ company_name_raw: '', search_names: '', domains: '' })
  const [overrideLoading, setOverrideLoading] = useState(false)
  const [bootstrapLoading, setBootstrapLoading] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setCoDebouncedSearch(coSearch), 300)
    return () => clearTimeout(t)
  }, [coSearch])

  // ---- Data fetching ----
  const { data: stats, loading: statsLoading } = useApiData(
    fetchProspectStats,
    [],
  )

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchProspects({
        company: debouncedSearch || undefined,
        status: statusFilter || undefined,
        seniority: seniorityFilter || undefined,
        limit: 200,
      }),
    [debouncedSearch, statusFilter, seniorityFilter],
  )

  const { data: mqData, loading: mqLoading, error: mqError, refresh: mqRefresh, refreshing: mqRefreshing } = useApiData(
    () => fetchManualQueue({ company: mqDebouncedSearch || undefined, limit: 200 }),
    [mqDebouncedSearch],
  )

  const { data: coData, loading: coLoading, error: coError, refresh: coRefresh, refreshing: coRefreshing } = useApiData(
    () => fetchCompanyOverrides({ company: coDebouncedSearch || undefined }),
    [coDebouncedSearch],
  )

  const prospects = data?.prospects ?? []
  const queueEntries = mqData?.queue ?? []
  const overrides = coData?.overrides ?? []

  // ---- Prospects columns ----
  const prospectColumns: Column<Prospect>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => (
        <div>
          <span className="text-white font-medium">{r.company_name || '--'}</span>
          {r.company_domain && (
            <span className="block text-xs text-slate-500">{r.company_domain}</span>
          )}
        </div>
      ),
      sortable: true,
      sortValue: (r) => r.company_name || '',
    },
    {
      key: 'churning_from',
      header: 'Churning From',
      render: (r) =>
        r.churning_from ? (
          <span className="text-amber-400 text-sm">{r.churning_from}</span>
        ) : (
          <span className="text-xs text-slate-500">--</span>
        ),
      sortable: true,
      sortValue: (r) => r.churning_from || '',
    },
    {
      key: 'name',
      header: 'Name',
      render: (r) => (
        <span className="text-slate-200">
          {[r.first_name, r.last_name].filter(Boolean).join(' ') || '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => `${r.first_name || ''} ${r.last_name || ''}`,
    },
    {
      key: 'title',
      header: 'Title',
      render: (r) => (
        <span className="text-sm text-slate-300 line-clamp-1 max-w-[200px]">
          {r.title || '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.title || '',
    },
    {
      key: 'location',
      header: 'Location',
      render: (r) => {
        const parts = [r.city, r.state].filter(Boolean)
        return parts.length > 0
          ? <span className="text-sm text-slate-300">{parts.join(', ')}</span>
          : <span className="text-xs text-slate-500">--</span>
      },
      sortable: true,
      sortValue: (r) => [r.city, r.state].filter(Boolean).join(', '),
    },
    {
      key: 'email',
      header: 'Email',
      render: (r) => (
        <span className="text-sm text-slate-400 font-mono">{r.email || '--'}</span>
      ),
    },
    {
      key: 'linkedin',
      header: '',
      render: (r) => r.linkedin_url ? (
        <a
          href={r.linkedin_url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center justify-center p-1 rounded text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/10 transition-colors"
          title="View LinkedIn profile"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      ) : null,
    },
    {
      key: 'seniority',
      header: 'Seniority',
      render: (r) => <SeniorityBadge seniority={r.seniority} />,
      sortable: true,
      sortValue: (r) => r.seniority || '',
    },
    {
      key: 'status',
      header: 'Status',
      render: (r) => <ProspectStatusBadge status={r.status} />,
      sortable: true,
      sortValue: (r) => r.status,
    },
    {
      key: 'sequence',
      header: 'Sequence',
      render: (r) => {
        if (!r.related_sequence_status) return <span className="text-xs text-slate-500">--</span>
        return (
          <div className="flex items-center gap-1.5">
            <SequenceStatusBadge status={r.related_sequence_status} />
            {r.related_sequence_current_step != null && r.related_sequence_max_steps != null && (
              <span className="text-xs text-slate-400">
                Step {r.related_sequence_current_step}/{r.related_sequence_max_steps}
              </span>
            )}
          </div>
        )
      },
      sortable: true,
      sortValue: (r) => r.related_sequence_status || '',
    },
    {
      key: 'email_status',
      header: 'Email Valid',
      render: (r) => <EmailStatusBadge status={r.email_status} />,
    },
    {
      key: 'created_at',
      header: 'Created',
      render: (r) => (
        <span className="text-xs text-slate-400">
          {r.created_at ? new Date(r.created_at).toLocaleDateString() : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.created_at || '',
    },
  ]

  // ---- Manual Queue columns ----
  const mqColumns: Column<ManualQueueEntry>[] = [
    {
      key: 'company_name_raw',
      header: 'Company (raw)',
      render: (r) => <span className="text-white font-medium">{r.company_name_raw}</span>,
      sortable: true,
      sortValue: (r) => r.company_name_raw,
    },
    {
      key: 'company_name_norm',
      header: 'Normalized',
      render: (r) => (
        <span className="text-sm text-slate-300">{r.company_name_norm || '--'}</span>
      ),
    },
    {
      key: 'domain',
      header: 'Domain',
      render: (r) => (
        <span className="text-sm text-cyan-400 font-mono">{r.domain || '--'}</span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (r) => <QueueStatusBadge status={r.status} />,
      sortable: true,
      sortValue: (r) => r.status,
    },
    {
      key: 'error_detail',
      header: 'Error',
      render: (r) => <ErrorDetailCell detail={r.error_detail} />,
    },
    {
      key: 'updated_at',
      header: 'Updated',
      render: (r) => (
        <span className="text-xs text-slate-400">
          {r.updated_at ? new Date(r.updated_at).toLocaleDateString() : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.updated_at || '',
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (r) => {
        if (r.status !== 'manual_review') return null
        if (resolvingId === r.id) {
          return (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={resolveDomain}
                onChange={(e) => setResolveDomain(e.target.value)}
                placeholder="domain.com"
                className="px-2 py-1 bg-slate-800/50 border border-slate-700/50 rounded text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-32"
              />
              <button
                onClick={() => handleResolve(r.id, 'retry')}
                disabled={resolveLoading}
                className="text-xs px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 disabled:opacity-50"
              >
                {resolveLoading ? '...' : 'Retry'}
              </button>
              <button
                onClick={() => { setResolvingId(null); setResolveDomain('') }}
                className="text-slate-400 hover:text-white"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          )
        }
        return (
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setResolvingId(r.id); setResolveDomain(r.domain || '') }}
              className="text-xs px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30"
              title="Retry with domain"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
            <button
              onClick={() => handleResolve(r.id, 'dismiss')}
              disabled={resolveLoading}
              className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 disabled:opacity-50"
              title="Dismiss"
            >
              <XCircle className="h-3 w-3" />
            </button>
          </div>
        )
      },
    },
  ]

  // ---- Company Overrides columns ----
  const coColumns: Column<CompanyOverride>[] = [
    {
      key: 'company_name_raw',
      header: 'Company (raw)',
      render: (r) => <span className="text-white font-medium">{r.company_name_raw}</span>,
      sortable: true,
      sortValue: (r) => r.company_name_raw,
    },
    {
      key: 'company_name_norm',
      header: 'Normalized',
      render: (r) => (
        <span className="text-sm text-slate-300">{r.company_name_norm}</span>
      ),
    },
    {
      key: 'search_names',
      header: 'Search Names',
      render: (r) => (
        <span className="text-sm text-slate-300">{r.search_names.length > 0 ? r.search_names.join(', ') : '--'}</span>
      ),
    },
    {
      key: 'domains',
      header: 'Domains',
      render: (r) => (
        <span className="text-sm text-cyan-400 font-mono">{r.domains.length > 0 ? r.domains.join(', ') : '--'}</span>
      ),
    },
    {
      key: 'updated_at',
      header: 'Updated',
      render: (r) => (
        <span className="text-xs text-slate-400">
          {r.updated_at ? new Date(r.updated_at).toLocaleDateString() : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.updated_at || '',
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (r) => {
        const overrideId = r.id
        if (!overrideId) return <span className="text-xs text-slate-500">settings</span>
        return (
          <div className="flex items-center gap-2">
            <button
              onClick={() => startEditOverride(r)}
              disabled={deletingId === overrideId}
              className="text-slate-400 hover:text-cyan-400 disabled:opacity-50"
              title="Edit"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => handleDeleteOverride(overrideId)}
              disabled={deletingId === overrideId}
              className="text-slate-400 hover:text-red-400 disabled:opacity-50"
              title="Delete"
            >
              {deletingId === overrideId ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
            </button>
          </div>
        )
      },
    },
  ]

  // ---- Handlers ----
  const handleResolve = async (id: string, action: 'retry' | 'dismiss') => {
    setResolveLoading(true)
    try {
      await resolveManualQueueEntry(id, {
        action,
        domain: action === 'retry' && resolveDomain ? resolveDomain : undefined,
      })
      setIsActionError(false)
      setActionResult(action === 'retry' ? 'Entry queued for retry' : 'Entry dismissed')
      setResolvingId(null)
      setResolveDomain('')
      mqRefresh()
    } catch (err) {
      setIsActionError(true)
      setActionResult(err instanceof Error ? err.message : 'Resolve failed')
    } finally {
      setResolveLoading(false)
    }
  }

  const startEditOverride = (r: CompanyOverride) => {
    if (!r.id) return
    setEditingOverrideId(r.id)
    setOverrideForm({
      company_name_raw: r.company_name_raw,
      search_names: r.search_names.join(', '),
      domains: r.domains.join(', '),
    })
    setShowOverrideForm(true)
  }

  const handleSaveOverride = async () => {
    if (!overrideForm.company_name_raw.trim()) return
    setOverrideLoading(true)
    try {
      const searchNames = overrideForm.search_names
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      const domains = overrideForm.domains
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      await upsertCompanyOverride({
        company_name_raw: overrideForm.company_name_raw.trim(),
        search_names: searchNames.length > 0 ? searchNames : undefined,
        domains: domains.length > 0 ? domains : undefined,
      })
      setIsActionError(false)
      setActionResult(editingOverrideId ? 'Override updated' : 'Override created')
      resetOverrideForm()
      coRefresh()
    } catch (err) {
      setIsActionError(true)
      setActionResult(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setOverrideLoading(false)
    }
  }

  const handleDeleteOverride = async (id: string) => {
    if (!confirm('Delete this company override?')) return
    setDeletingId(id)
    try {
      await deleteCompanyOverride(id)
      setIsActionError(false)
      setActionResult('Override deleted')
      coRefresh()
    } catch (err) {
      setIsActionError(true)
      setActionResult(err instanceof Error ? err.message : 'Delete failed')
    } finally {
      setDeletingId(null)
    }
  }

  const handleBootstrap = async () => {
    setBootstrapLoading(true)
    try {
      const result = await bootstrapCompanyOverrides()
      setIsActionError(false)
      setActionResult(`Bootstrapped ${result.imported} override(s) from settings`)
      coRefresh()
    } catch (err) {
      setIsActionError(true)
      setActionResult(err instanceof Error ? err.message : 'Bootstrap failed')
    } finally {
      setBootstrapLoading(false)
    }
  }

  const resetOverrideForm = () => {
    setShowOverrideForm(false)
    setEditingOverrideId(null)
    setOverrideForm({ company_name_raw: '', search_names: '', domains: '' })
  }

  const handleExportCsv = () => {
    downloadProspectsCsv({
      company: debouncedSearch || undefined,
      status: statusFilter || undefined,
      seniority: seniorityFilter || undefined,
    })
  }

  const hasFilters = !!companySearch || !!statusFilter || !!seniorityFilter
  const clearFilters = () => {
    setCompanySearch('')
    setStatusFilter('')
    setSeniorityFilter('')
  }

  function tabLabel(tab: ProspectsTab): string {
    switch (tab) {
      case 'prospects':
        return 'All Prospects' + (data?.count != null ? ' (' + data.count + ')' : '')
      case 'manual_queue':
        return 'Manual Queue' + (mqData?.count != null ? ' (' + mqData.count + ')' : '')
      case 'company_overrides':
        return 'Company Overrides' + (coData?.count != null ? ' (' + coData.count + ')' : '')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Users className="h-6 w-6 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Prospects</h1>
        </div>
        <div className="flex items-center gap-2">
          {activeTab === 'prospects' && (
            <button
              onClick={handleExportCsv}
              className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
            >
              <Download className="h-4 w-4" />
              Export CSV
            </button>
          )}
          <button
            onClick={() => {
              if (activeTab === 'prospects') refresh()
              else if (activeTab === 'manual_queue') mqRefresh()
              else coRefresh()
            }}
            disabled={activeTab === 'prospects' ? refreshing : activeTab === 'manual_queue' ? mqRefreshing : coRefreshing}
            className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
          >
            <RefreshCw className={clsx('h-4 w-4', (activeTab === 'prospects' ? refreshing : activeTab === 'manual_queue' ? mqRefreshing : coRefreshing) && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total Prospects"
          value={stats?.total ?? 0}
          icon={<Users className="h-4 w-4" />}
          skeleton={statsLoading}
        />
        <StatCard
          label="Active"
          value={stats?.active ?? 0}
          icon={<UserCheck className="h-4 w-4" />}
          skeleton={statsLoading}
        />
        <StatCard
          label="Contacted"
          value={stats?.contacted ?? 0}
          icon={<Mail className="h-4 w-4" />}
          skeleton={statsLoading}
        />
        <StatCard
          label="This Month"
          value={stats?.this_month ?? 0}
          icon={<UserPlus className="h-4 w-4" />}
          skeleton={statsLoading}
          sub="Last 30 days"
        />
      </div>

      {/* Action result banner */}
      {actionResult && (
        <div className={clsx(
          'rounded-lg p-3 flex items-center justify-between',
          isActionError
            ? 'bg-red-500/10 border border-red-500/30'
            : 'bg-cyan-500/10 border border-cyan-500/30',
        )}>
          <span className={clsx('text-sm', isActionError ? 'text-red-400' : 'text-cyan-400')}>{actionResult}</span>
          <button onClick={() => setActionResult(null)} className={clsx(isActionError ? 'text-red-400' : 'text-cyan-400', 'hover:text-white')}>
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-slate-700/50">
        {(['prospects', 'manual_queue', 'company_overrides'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => { setActiveTab(tab); setActionResult(null); setResolvingId(null); setResolveDomain(''); resetOverrideForm() }}
            className={clsx(
              'px-4 py-2 text-sm font-medium transition-colors border-b-2',
              activeTab === tab
                ? 'text-cyan-400 border-cyan-400'
                : 'text-slate-400 border-transparent hover:text-white',
            )}
          >
            {tabLabel(tab)}
          </button>
        ))}
      </div>

      {/* ================================================================= */}
      {/* Prospects tab                                                      */}
      {/* ================================================================= */}
      {activeTab === 'prospects' && (
        <>
          {/* Filters */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                type="text"
                value={companySearch}
                onChange={(e) => setCompanySearch(e.target.value)}
                placeholder="Search company..."
                className="pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-56"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-cyan-500/50"
            >
              <option value="">All Statuses</option>
              <option value="active">Active</option>
              <option value="contacted">Contacted</option>
              <option value="opted_out">Opted Out</option>
              <option value="bounced">Bounced</option>
              <option value="suppressed">Suppressed</option>
            </select>
            <select
              value={seniorityFilter}
              onChange={(e) => setSeniorityFilter(e.target.value)}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-cyan-500/50"
            >
              <option value="">All Seniority</option>
              <option value="c_suite">C-Suite</option>
              <option value="vp">VP</option>
              <option value="director">Director</option>
              <option value="manager">Manager</option>
              <option value="senior">Senior</option>
            </select>
            {hasFilters && (
              <button
                onClick={clearFilters}
                className="text-xs text-slate-400 hover:text-white"
              >
                Clear filters
              </button>
            )}
          </div>

          {/* Count */}
          <div className="text-sm text-slate-400">
            {loading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-3 w-3 animate-spin" /> Loading...
              </span>
            ) : (
              <span>{data?.count ?? 0} prospects found</span>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 text-sm">
              {error.message}
            </div>
          )}

          {/* Table */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
            {loading ? (
              <DataTable columns={prospectColumns} data={[]} skeletonRows={8} />
            ) : (
              <DataTable
                columns={prospectColumns}
                data={prospects}
                onRowClick={setViewingProspect}
                emptyMessage="No prospects match your filters"
                emptyAction={hasFilters ? { label: 'Clear all filters', onClick: clearFilters } : undefined}
              />
            )}
          </div>
        </>
      )}

      {/* ================================================================= */}
      {/* Manual Queue tab                                                   */}
      {/* ================================================================= */}
      {activeTab === 'manual_queue' && (
        <>
          {/* Search */}
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                type="text"
                value={mqSearch}
                onChange={(e) => setMqSearch(e.target.value)}
                placeholder="Search company..."
                className="pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-56"
              />
            </div>
          </div>

          {/* Count */}
          <div className="text-sm text-slate-400">
            {mqLoading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-3 w-3 animate-spin" /> Loading...
              </span>
            ) : (
              <span>{mqData?.count ?? 0} queue entries</span>
            )}
          </div>

          {/* Error */}
          {mqError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 text-sm">
              {mqError.message}
            </div>
          )}

          {/* Table */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
            {mqLoading ? (
              <DataTable columns={mqColumns} data={[]} skeletonRows={8} />
            ) : (
              <DataTable
                columns={mqColumns}
                data={queueEntries}
                emptyMessage="No entries in the manual queue"
              />
            )}
          </div>
        </>
      )}

      {/* ================================================================= */}
      {/* Company Overrides tab                                              */}
      {/* ================================================================= */}
      {activeTab === 'company_overrides' && (
        <>
          {/* Search + actions */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                type="text"
                value={coSearch}
                onChange={(e) => setCoSearch(e.target.value)}
                placeholder="Search company..."
                className="pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-56"
              />
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => { resetOverrideForm(); setShowOverrideForm(true) }}
                className="flex items-center gap-2 px-3 py-2 text-sm text-cyan-400 bg-cyan-500/10 rounded-lg hover:bg-cyan-500/20 transition-colors"
              >
                <Plus className="h-4 w-4" />
                Add Override
              </button>
              <button
                onClick={handleBootstrap}
                disabled={bootstrapLoading}
                className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors disabled:opacity-50"
              >
                {bootstrapLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Building2 className="h-4 w-4" />}
                {bootstrapLoading ? 'Bootstrapping...' : 'Bootstrap from Settings'}
              </button>
            </div>
          </div>

          {/* Inline form */}
          {showOverrideForm && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-white">
                  {editingOverrideId ? 'Edit Override' : 'New Override'}
                </span>
                <button onClick={resetOverrideForm} className="text-slate-400 hover:text-white">
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Company Name (required)</label>
                  <input
                    type="text"
                    value={overrideForm.company_name_raw}
                    onChange={(e) => setOverrideForm((f) => ({ ...f, company_name_raw: e.target.value }))}
                    disabled={Boolean(editingOverrideId)}
                    className={clsx(
                      'w-full px-3 py-2 bg-slate-900/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50',
                      editingOverrideId && 'cursor-not-allowed bg-slate-900/30 text-slate-500 focus:border-slate-700/50'
                    )}
                    placeholder="Acme Corp"
                  />
                  {editingOverrideId && (
                    <p className="mt-1 text-[11px] text-slate-500">
                      Company name is locked while editing. Create a new override to rename the key.
                    </p>
                  )}
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Search Names (comma-separated)</label>
                  <input
                    type="text"
                    value={overrideForm.search_names}
                    onChange={(e) => setOverrideForm((f) => ({ ...f, search_names: e.target.value }))}
                    className="w-full px-3 py-2 bg-slate-900/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                    placeholder="Acme, Acme Corporation"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Domains (comma-separated)</label>
                  <input
                    type="text"
                    value={overrideForm.domains}
                    onChange={(e) => setOverrideForm((f) => ({ ...f, domains: e.target.value }))}
                    className="w-full px-3 py-2 bg-slate-900/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                    placeholder="acme.com, acmecorp.com"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <button
                  onClick={resetOverrideForm}
                  className="px-3 py-2 text-sm text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveOverride}
                  disabled={overrideLoading || !overrideForm.company_name_raw.trim()}
                  className="px-4 py-2 text-sm text-white bg-cyan-600 rounded-lg hover:bg-cyan-500 disabled:opacity-50 transition-colors"
                >
                  {overrideLoading ? 'Saving...' : editingOverrideId ? 'Update' : 'Create'}
                </button>
              </div>
            </div>
          )}

          {/* Count */}
          <div className="text-sm text-slate-400">
            {coLoading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-3 w-3 animate-spin" /> Loading...
              </span>
            ) : (
              <span>{coData?.count ?? 0} overrides</span>
            )}
          </div>

          {/* Error */}
          {coError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 text-sm">
              {coError.message}
            </div>
          )}

          {/* Table */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
            {coLoading ? (
              <DataTable columns={coColumns} data={[]} skeletonRows={8} />
            ) : (
              <DataTable
                columns={coColumns}
                data={overrides}
                emptyMessage="No company overrides configured"
              />
            )}
          </div>
        </>
      )}

      {/* Prospect Detail Drawer */}
      {viewingProspect && (
        <ProspectDetailDrawer
          prospect={viewingProspect}
          onClose={() => setViewingProspect(null)}
        />
      )}
    </div>
  )
}
