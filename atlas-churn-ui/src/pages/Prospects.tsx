import { useState, useEffect } from 'react'
import {
  Users,
  RefreshCw,
  Search,
  UserCheck,
  UserPlus,
  Mail,
  Loader2,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '../hooks/useApiData'
import DataTable from '../components/DataTable'
import StatCard from '../components/StatCard'
import type { Column } from '../components/DataTable'
import type { Prospect } from '../types'
import { fetchProspects, fetchProspectStats } from '../api/client'

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

function SequenceStatusBadge({ status }: { status: string | null | undefined }) {
  if (!status) return <span className="text-xs text-slate-500">--</span>
  const styles: Record<string, string> = {
    active: 'bg-cyan-500/20 text-cyan-400',
    paused: 'bg-amber-500/20 text-amber-400',
    completed: 'bg-green-500/20 text-green-400',
    replied: 'bg-green-500/20 text-green-400',
    bounced: 'bg-red-500/20 text-red-400',
    unsubscribed: 'bg-slate-500/20 text-slate-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', styles[status] || 'bg-slate-500/20 text-slate-400')}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}

export default function ProspectsPage() {
  const [companySearch, setCompanySearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [seniorityFilter, setSeniorityFilter] = useState('')

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(companySearch), 300)
    return () => clearTimeout(t)
  }, [companySearch])

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

  const prospects = data?.prospects ?? []

  const columns: Column<Prospect>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => (
        <div className="min-w-0">
          <span className="text-white font-medium">{r.company_name || '--'}</span>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {r.company_domain && (
              <span className="text-xs text-slate-500">{r.company_domain}</span>
            )}
            {r.churning_from && (
              <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[10px] font-medium text-red-300">
                from {r.churning_from}
              </span>
            )}
            {r.target_persona && (
              <span className="rounded-full bg-slate-700/50 px-2 py-0.5 text-[10px] font-medium text-slate-300">
                {r.target_persona}
              </span>
            )}
          </div>
        </div>
      ),
      sortable: true,
      sortValue: (r) => r.company_name || '',
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
      key: 'signal',
      header: 'Buying Signal',
      render: (r) => {
        const signal = r.reasoning_atom_context?.account_signals?.[0]
        if (!signal) return <span className="text-xs text-slate-500">--</span>
        return (
          <div className="min-w-0 max-w-[260px] text-xs">
            <p className="text-slate-200 line-clamp-1">
              {[signal.buying_stage, signal.primary_pain].filter(Boolean).join(' | ')}
            </p>
            <p className="text-slate-400 line-clamp-1">
              {[signal.competitor_context, signal.contract_end || signal.decision_timeline].filter(Boolean).join(' | ')}
            </p>
            {signal.quote && (
              <p className="mt-1 text-[11px] italic text-slate-500 line-clamp-2">
                "{signal.quote}"
              </p>
            )}
          </div>
        )
      },
    },
    {
      key: 'email',
      header: 'Email',
      render: (r) => (
        <span className="text-sm text-slate-400 font-mono">{r.email || '--'}</span>
      ),
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
      render: (r) => (
        <div className="min-w-0 text-xs">
          <SequenceStatusBadge status={r.related_sequence_status} />
          {(r.related_sequence_current_step != null || r.related_sequence_last_sent_at) && (
            <div className="mt-1 text-slate-500">
              {r.related_sequence_current_step != null && (
                <span>
                  step {r.related_sequence_current_step}
                  {r.related_sequence_max_steps ? `/${r.related_sequence_max_steps}` : ''}
                </span>
              )}
              {r.related_sequence_last_sent_at && (
                <span className="block">
                  sent {new Date(r.related_sequence_last_sent_at).toLocaleDateString()}
                </span>
              )}
            </div>
          )}
        </div>
      ),
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

  const hasFilters = !!companySearch || !!statusFilter || !!seniorityFilter
  const clearFilters = () => {
    setCompanySearch('')
    setStatusFilter('')
    setSeniorityFilter('')
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Users className="h-6 w-6 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Prospects</h1>
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
          <DataTable columns={columns} data={[]} skeletonRows={8} />
        ) : (
          <DataTable
            columns={columns}
            data={prospects}
            emptyMessage="No prospects match your filters"
            emptyAction={hasFilters ? { label: 'Clear all filters', onClick: clearFilters } : undefined}
          />
        )}
      </div>
    </div>
  )
}
