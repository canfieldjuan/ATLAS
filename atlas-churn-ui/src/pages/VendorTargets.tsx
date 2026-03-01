import { useState, useEffect } from 'react'
import {
  Shield,
  Plus,
  Pencil,
  Trash2,
  X,
  RefreshCw,
  FileBarChart,
  Send,
  Building2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  fetchVendorTargets,
  createVendorTarget,
  updateVendorTarget,
  deleteVendorTarget,
  generateVendorReport,
  generateCampaigns,
} from '../api/client'
import type { VendorTarget } from '../types'

const TARGET_MODES = [
  { value: 'vendor_retention', label: 'Vendor Retention' },
  { value: 'challenger_intel', label: 'Challenger Intel' },
]

const TIERS = ['report', 'dashboard', 'api']

function ModeBadge({ mode }: { mode: string }) {
  const colors: Record<string, string> = {
    vendor_retention: 'bg-amber-500/20 text-amber-400',
    challenger_intel: 'bg-purple-500/20 text-purple-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', colors[mode] ?? 'bg-slate-500/20 text-slate-400')}>
      {mode.replace(/_/g, ' ')}
    </span>
  )
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, string> = {
    report: 'bg-slate-500/20 text-slate-400',
    dashboard: 'bg-cyan-500/20 text-cyan-400',
    api: 'bg-green-500/20 text-green-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', colors[tier] ?? 'bg-slate-500/20 text-slate-400')}>
      {tier}
    </span>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: 'bg-green-500/20 text-green-400',
    paused: 'bg-amber-500/20 text-amber-400',
    inactive: 'bg-slate-500/20 text-slate-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', colors[status] ?? 'bg-slate-500/20 text-slate-400')}>
      {status}
    </span>
  )
}

const EMPTY_FORM = {
  company_name: '',
  target_mode: 'vendor_retention' as string,
  contact_name: '',
  contact_email: '',
  contact_role: '',
  products_tracked: [] as string[],
  competitors_tracked: [] as string[],
  tier: 'report',
  status: 'active',
  notes: '',
}

export default function VendorTargets() {
  const [modeFilter, setModeFilter] = useState<string>('')
  const [searchInput, setSearchInput] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')

  // Form state
  const [showForm, setShowForm] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [productsInput, setProductsInput] = useState('')
  const [competitorsInput, setCompetitorsInput] = useState('')
  const [saving, setSaving] = useState(false)

  // Detail panel
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [generatingReport, setGeneratingReport] = useState<string | null>(null)
  const [generatingCampaign, setGeneratingCampaign] = useState<string | null>(null)
  const [actionResult, setActionResult] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(searchInput), 300)
    return () => clearTimeout(t)
  }, [searchInput])

  const { data, loading, error, refresh, refreshing } = useApiData(
    async () => {
      const res = await fetchVendorTargets({
        target_mode: modeFilter || undefined,
        search: debouncedSearch || undefined,
        limit: 200,
      })
      return res
    },
    [modeFilter, debouncedSearch],
  )

  const targets = data?.targets ?? []
  const vendorTargets = targets.filter(t => t.target_mode === 'vendor_retention')
  const challengerTargets = targets.filter(t => t.target_mode === 'challenger_intel')

  function openAdd(mode: string = 'vendor_retention') {
    setEditingId(null)
    setForm({ ...EMPTY_FORM, target_mode: mode })
    setProductsInput('')
    setCompetitorsInput('')
    setShowForm(true)
  }

  function openEdit(t: VendorTarget) {
    setEditingId(t.id)
    setForm({
      company_name: t.company_name,
      target_mode: t.target_mode,
      contact_name: t.contact_name ?? '',
      contact_email: t.contact_email ?? '',
      contact_role: t.contact_role ?? '',
      products_tracked: t.products_tracked ?? [],
      competitors_tracked: t.competitors_tracked ?? [],
      tier: t.tier,
      status: t.status,
      notes: t.notes ?? '',
    })
    setProductsInput((t.products_tracked ?? []).join(', '))
    setCompetitorsInput((t.competitors_tracked ?? []).join(', '))
    setShowForm(true)
  }

  async function handleSave() {
    setSaving(true)
    const products = productsInput.split(',').map(s => s.trim()).filter(Boolean)
    const competitors = competitorsInput.split(',').map(s => s.trim()).filter(Boolean)
    const payload = {
      ...form,
      products_tracked: products.length ? products : null,
      competitors_tracked: competitors.length ? competitors : null,
      notes: form.notes || null,
    }
    try {
      if (editingId) {
        await updateVendorTarget(editingId, payload)
      } else {
        await createVendorTarget(payload)
      }
      setShowForm(false)
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(id: string) {
    if (!confirm('Delete this target?')) return
    try {
      await deleteVendorTarget(id)
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Delete failed')
    }
  }

  async function handleGenerateReport(t: VendorTarget) {
    setGeneratingReport(t.id)
    setActionResult(null)
    try {
      const report = await generateVendorReport(t.id)
      setActionResult(
        `Report generated: ${report.signal_count} signals, ${report.high_urgency_count} critical for ${t.company_name}`,
      )
      refresh()
    } catch (err) {
      setActionResult(err instanceof Error ? err.message : 'Report generation failed')
    } finally {
      setGeneratingReport(null)
    }
  }

  async function handleGenerateCampaign(t: VendorTarget) {
    setGeneratingCampaign(t.id)
    setActionResult(null)
    try {
      const result = await generateCampaigns({
        vendor_name: t.company_name,
        target_mode: t.target_mode,
        min_score: 50,
        limit: 5,
      })
      setActionResult(
        `Generated ${result.generated ?? 0} campaign(s) for ${t.company_name}`,
      )
      refresh()
    } catch (err) {
      setActionResult(err instanceof Error ? err.message : 'Campaign generation failed')
    } finally {
      setGeneratingCampaign(null)
    }
  }

  const columns: Column<VendorTarget>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-white font-medium">{r.company_name}</span>,
    },
    {
      key: 'mode',
      header: 'Mode',
      render: (r) => <ModeBadge mode={r.target_mode} />,
    },
    {
      key: 'contact',
      header: 'Contact',
      render: (r) => (
        <div>
          <span className="text-slate-300 text-xs">{r.contact_name ?? '--'}</span>
          {r.contact_role && (
            <span className="text-slate-500 text-xs ml-1">({r.contact_role})</span>
          )}
        </div>
      ),
    },
    {
      key: 'tier',
      header: 'Tier',
      render: (r) => <TierBadge tier={r.tier} />,
    },
    {
      key: 'products',
      header: 'Products',
      render: (r) => (
        <span className="text-slate-400 text-xs">
          {(r.products_tracked ?? []).join(', ') || '--'}
        </span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (r) => <StatusBadge status={r.status} />,
    },
    {
      key: 'actions',
      header: '',
      render: (r) => (
        <div className="flex items-center gap-1">
          {r.target_mode === 'vendor_retention' && (
            <button
              onClick={(e) => { e.stopPropagation(); handleGenerateReport(r) }}
              disabled={generatingReport === r.id}
              className="p-1 text-slate-400 hover:text-cyan-400 transition-colors disabled:opacity-50"
              title="Generate Report"
            >
              <FileBarChart className={clsx('h-3.5 w-3.5', generatingReport === r.id && 'animate-pulse')} />
            </button>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); handleGenerateCampaign(r) }}
            disabled={generatingCampaign === r.id}
            className="p-1 text-slate-400 hover:text-green-400 transition-colors disabled:opacity-50"
            title="Generate Campaign"
          >
            <Send className={clsx('h-3.5 w-3.5', generatingCampaign === r.id && 'animate-pulse')} />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); openEdit(r) }}
            className="p-1 text-slate-400 hover:text-white transition-colors"
            title="Edit"
          >
            <Pencil className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); handleDelete(r.id) }}
            className="p-1 text-slate-400 hover:text-red-400 transition-colors"
            title="Delete"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Vendor Targets</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => openAdd('vendor_retention')}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 transition-colors"
          >
            <Plus className="h-4 w-4" />
            Add Vendor
          </button>
          <button
            onClick={() => openAdd('challenger_intel')}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-colors"
          >
            <Plus className="h-4 w-4" />
            Add Challenger
          </button>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Action result banner */}
      {actionResult && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 flex items-center justify-between">
          <span className="text-sm text-cyan-400">{actionResult}</span>
          <button onClick={() => setActionResult(null)} className="text-cyan-400 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Targets"
          value={targets.length}
          icon={<Shield className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Vendor Retention"
          value={vendorTargets.length}
          icon={<Building2 className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Challenger Intel"
          value={challengerTargets.length}
          icon={<Building2 className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Active"
          value={targets.filter(t => t.status === 'active').length}
          icon={<Shield className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search company..."
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-48"
        />
        <select
          value={modeFilter}
          onChange={(e) => setModeFilter(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-cyan-500/50"
        >
          <option value="">All modes</option>
          {TARGET_MODES.map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
      </div>

      {/* Inline Form */}
      {showForm && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-white">
              {editingId ? 'Edit Target' : 'New Target'}
            </span>
            <button onClick={() => setShowForm(false)} className="text-slate-400 hover:text-white">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            <input
              placeholder="Company name *"
              value={form.company_name}
              onChange={(e) => setForm(f => ({ ...f, company_name: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            <select
              value={form.target_mode}
              onChange={(e) => setForm(f => ({ ...f, target_mode: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-cyan-500/50"
            >
              {TARGET_MODES.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <input
              placeholder="Contact name"
              value={form.contact_name}
              onChange={(e) => setForm(f => ({ ...f, contact_name: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            <input
              placeholder="Contact email"
              value={form.contact_email}
              onChange={(e) => setForm(f => ({ ...f, contact_email: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            <input
              placeholder="Contact role (e.g., VP Customer Success)"
              value={form.contact_role}
              onChange={(e) => setForm(f => ({ ...f, contact_role: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            <select
              value={form.tier}
              onChange={(e) => setForm(f => ({ ...f, tier: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-cyan-500/50"
            >
              {TIERS.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <input
              placeholder="Products tracked (comma-separated)"
              value={productsInput}
              onChange={(e) => setProductsInput(e.target.value)}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            <input
              placeholder="Competitors tracked (comma-separated)"
              value={competitorsInput}
              onChange={(e) => setCompetitorsInput(e.target.value)}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            <input
              placeholder="Notes"
              value={form.notes}
              onChange={(e) => setForm(f => ({ ...f, notes: e.target.value }))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <div className="flex items-center gap-3 pt-1">
            <button
              onClick={handleSave}
              disabled={saving || !form.company_name}
              className="px-4 py-1.5 rounded-lg text-sm bg-cyan-500 text-white hover:bg-cyan-600 transition-colors disabled:opacity-50"
            >
              {saving ? 'Saving...' : editingId ? 'Update' : 'Create'}
            </button>
            <button
              onClick={() => setShowForm(false)}
              className="px-4 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Targets Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          All Targets ({targets.length})
        </h3>
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={columns}
            data={targets}
            emptyMessage="No vendor targets configured"
            emptyAction={{
              label: 'Add Target',
              onClick: () => openAdd(),
            }}
          />
        )}
      </div>
    </div>
  )
}
