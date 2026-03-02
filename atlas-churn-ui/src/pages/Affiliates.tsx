import { useState, useEffect } from 'react'
import {
  Handshake,
  Target,
  MousePointerClick,
  TrendingUp,
  RefreshCw,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Plus,
  Pencil,
  Trash2,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  fetchAffiliateOpportunities,
  fetchAffiliatePartners,
  fetchClickSummary,
  createAffiliatePartner,
  updateAffiliatePartner,
  deleteAffiliatePartner,
  recordAffiliateClick,
} from '../api/client'
import type { AffiliateOpportunity, AffiliatePartner, ClickSummary } from '../types'

interface AffiliatesData {
  opportunities: AffiliateOpportunity[]
  partners: AffiliatePartner[]
  clicks: ClickSummary[]
}

function ScoreBadge({ score }: { score: number }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        score >= 80
          ? 'bg-green-500/20 text-green-400'
          : score >= 60
            ? 'bg-cyan-500/20 text-cyan-400'
            : score >= 40
              ? 'bg-amber-500/20 text-amber-400'
              : 'bg-slate-500/20 text-slate-400',
      )}
    >
      {score}
    </span>
  )
}

const EMPTY_PARTNER = {
  name: '',
  product_name: '',
  product_aliases: [] as string[],
  category: '',
  affiliate_url: '',
  commission_type: 'unknown',
  commission_value: '',
  notes: '',
  enabled: true,
}

const COMMISSION_TYPES = ['cpa', 'recurring', 'rev_share', 'flat', 'unknown']

export default function Affiliates() {
  // Filters
  const [vendorSearch, setVendorSearch] = useState('')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [minUrgency, setMinUrgency] = useState(5)
  const [minScore, setMinScore] = useState(0)
  const [dmOnly, setDmOnly] = useState(false)

  // Partner management
  const [showPartners, setShowPartners] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [form, setForm] = useState(EMPTY_PARTNER)
  const [aliasInput, setAliasInput] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(t)
  }, [vendorSearch])

  const { data, loading, error, refresh, refreshing } = useApiData<AffiliatesData>(
    async () => {
      const [oppRes, partnerRes, clickRes] = await Promise.all([
        fetchAffiliateOpportunities({
          min_urgency: minUrgency,
          min_score: minScore,
          vendor_name: debouncedVendor || undefined,
          dm_only: dmOnly || undefined,
          limit: 100,
        }),
        fetchAffiliatePartners(),
        fetchClickSummary(),
      ])
      return {
        opportunities: oppRes.opportunities,
        partners: partnerRes.partners,
        clicks: clickRes.clicks,
      }
    },
    [minUrgency, minScore, debouncedVendor, dmOnly],
  )

  const opportunities = data?.opportunities ?? []
  const partners = data?.partners ?? []
  const clicks = data?.clicks ?? []

  const activePartners = partners.filter((p) => p.enabled).length
  const totalClicks = clicks.reduce((s, c) => s + c.click_count, 0)
  const avgScore =
    opportunities.length > 0
      ? Math.round(opportunities.reduce((s, o) => s + o.opportunity_score, 0) / opportunities.length)
      : 0

  // -- Click handler --
  async function handleLinkClick(opp: AffiliateOpportunity) {
    try {
      await recordAffiliateClick(opp.partner_id, opp.review_id)
    } catch {
      // fire-and-forget
    }
    window.open(opp.affiliate_url, '_blank', 'noopener')
  }

  // -- Partner form --
  function openAdd() {
    setEditingId(null)
    setForm(EMPTY_PARTNER)
    setAliasInput('')
    setShowForm(true)
  }

  function openEdit(p: AffiliatePartner) {
    setEditingId(p.id)
    setForm({
      name: p.name,
      product_name: p.product_name,
      product_aliases: p.product_aliases,
      category: p.category ?? '',
      affiliate_url: p.affiliate_url,
      commission_type: p.commission_type,
      commission_value: p.commission_value ?? '',
      notes: p.notes ?? '',
      enabled: p.enabled,
    })
    setAliasInput(p.product_aliases.join(', '))
    setShowForm(true)
  }

  async function handleSave() {
    setSaving(true)
    const aliases = aliasInput
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    const payload = {
      ...form,
      product_aliases: aliases,
      category: form.category || null,
      commission_value: form.commission_value || null,
      notes: form.notes || null,
    }
    try {
      if (editingId) {
        await updateAffiliatePartner(editingId, payload)
      } else {
        await createAffiliatePartner(payload)
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
    if (!confirm('Delete this affiliate partner?')) return
    try {
      await deleteAffiliatePartner(id)
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Delete failed')
    }
  }

  async function handleToggleEnabled(p: AffiliatePartner) {
    try {
      await updateAffiliatePartner(p.id, { enabled: !p.enabled })
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Toggle failed')
    }
  }

  // -- Table columns --
  const columns: Column<AffiliateOpportunity>[] = [
    {
      key: 'score',
      header: 'Score',
      render: (r) => <ScoreBadge score={r.opportunity_score} />,
      sortable: true,
      sortValue: (r) => r.opportunity_score,
    },
    {
      key: 'vendor',
      header: 'Churning From',
      render: (r) => <span className="text-white font-medium">{r.vendor_name}</span>,
    },
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-slate-300">{r.reviewer_company ?? '--'}</span>,
    },
    {
      key: 'competitor',
      header: 'Considering',
      render: (r) => (
        <span className="text-cyan-400 font-medium">{r.competitor_name}</span>
      ),
    },
    {
      key: 'context',
      header: 'Context',
      render: (r) => (
        <span className="text-slate-400 text-xs max-w-[200px] truncate block">
          {r.mention_context ?? '--'}
        </span>
      ),
    },
    {
      key: 'reason',
      header: 'Reason',
      render: (r) => (
        <span className="text-slate-400 text-xs max-w-[160px] truncate block">
          {r.mention_reason ?? '--'}
        </span>
      ),
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency} />,
      sortable: true,
      sortValue: (r) => r.urgency,
    },
    {
      key: 'dm',
      header: 'DM',
      render: (r) =>
        r.is_dm ? (
          <span className="text-cyan-400 text-xs font-medium">Yes</span>
        ) : (
          <span className="text-slate-500 text-xs">No</span>
        ),
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (r) => {
        if (!r.buying_stage || r.buying_stage === 'unknown')
          return <span className="text-slate-500 text-xs">--</span>
        const colors: Record<string, string> = {
          active_purchase: 'text-red-400',
          renewal_decision: 'text-amber-400',
          evaluation: 'text-cyan-400',
          post_purchase: 'text-slate-400',
        }
        return (
          <span className={`text-xs font-medium ${colors[r.buying_stage] ?? 'text-slate-400'}`}>
            {r.buying_stage.replace(/_/g, ' ')}
          </span>
        )
      },
    },
    {
      key: 'seats',
      header: 'Seats',
      render: (r) => <span className="text-slate-300">{r.seat_count ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.seat_count ?? 0,
    },
    {
      key: 'link',
      header: '',
      render: (r) => (
        <button
          onClick={(e) => {
            e.stopPropagation()
            handleLinkClick(r)
          }}
          className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-cyan-400 hover:bg-cyan-500/10 transition-colors"
          title={r.affiliate_url}
        >
          <ExternalLink className="h-3 w-3" />
          Link
        </button>
      ),
    },
  ]

  // -- Partner table columns --
  const partnerColumns: Column<AffiliatePartner>[] = [
    {
      key: 'name',
      header: 'Name',
      render: (r) => <span className="text-white font-medium">{r.name}</span>,
    },
    {
      key: 'product',
      header: 'Product',
      render: (r) => <span className="text-slate-300">{r.product_name}</span>,
    },
    {
      key: 'category',
      header: 'Category',
      render: (r) => <span className="text-slate-400">{r.category ?? '--'}</span>,
    },
    {
      key: 'commission',
      header: 'Commission',
      render: (r) => (
        <span className="text-slate-300 text-xs">
          {r.commission_type}{r.commission_value ? ` (${r.commission_value})` : ''}
        </span>
      ),
    },
    {
      key: 'enabled',
      header: 'Active',
      render: (r) => (
        <button
          onClick={(e) => {
            e.stopPropagation()
            handleToggleEnabled(r)
          }}
          className={clsx(
            'w-9 h-5 rounded-full relative transition-colors',
            r.enabled ? 'bg-cyan-500' : 'bg-slate-600',
          )}
        >
          <span
            className={clsx(
              'absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform',
              r.enabled ? 'left-[18px]' : 'left-0.5',
            )}
          />
        </button>
      ),
    },
    {
      key: 'actions',
      header: '',
      render: (r) => (
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation()
              openEdit(r)
            }}
            className="p-1 text-slate-400 hover:text-white transition-colors"
          >
            <Pencil className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation()
              handleDelete(r.id)
            }}
            className="p-1 text-slate-400 hover:text-red-400 transition-colors"
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
        <h1 className="text-2xl font-bold text-white">Affiliate Opportunities</h1>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Active Partners"
          value={activePartners}
          icon={<Handshake className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Opportunities Found"
          value={opportunities.length}
          icon={<Target className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Clicks (30d)"
          value={totalClicks}
          icon={<MousePointerClick className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Avg Score"
          value={avgScore}
          icon={<TrendingUp className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      {/* Opportunities Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Ranked Opportunities</h3>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <input
            type="text"
            placeholder="Filter vendor..."
            value={vendorSearch}
            onChange={(e) => setVendorSearch(e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-48"
          />
          <label className="flex items-center gap-2 text-sm text-slate-400">
            Min Urgency
            <input
              type="number"
              min={0}
              max={10}
              step={1}
              value={minUrgency}
              onChange={(e) => setMinUrgency(Number(e.target.value))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-2 py-1.5 text-sm text-white w-16 focus:outline-none focus:border-cyan-500/50"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-400">
            Min Score
            <input
              type="number"
              min={0}
              max={100}
              step={5}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-2 py-1.5 text-sm text-white w-16 focus:outline-none focus:border-cyan-500/50"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={dmOnly}
              onChange={(e) => setDmOnly(e.target.checked)}
              className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/30"
            />
            DM only
          </label>
        </div>

        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={columns}
            data={opportunities}
            emptyMessage="No affiliate opportunities found"
            emptyAction={{
              label: 'Add Partners',
              onClick: () => {
                setShowPartners(true)
                openAdd()
              },
            }}
          />
        )}
      </div>

      {/* Partner Management */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl">
        <button
          onClick={() => setShowPartners((v) => !v)}
          className="w-full flex items-center justify-between p-5 text-left"
        >
          <h3 className="text-sm font-medium text-slate-300">Partner Management</h3>
          {showPartners ? (
            <ChevronUp className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          )}
        </button>

        {showPartners && (
          <div className="px-5 pb-5 space-y-4">
            <div className="flex items-center justify-end">
              <button
                onClick={openAdd}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
              >
                <Plus className="h-4 w-4" />
                Add Partner
              </button>
            </div>

            {/* Inline Form */}
            {showForm && (
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-white">
                    {editingId ? 'Edit Partner' : 'New Partner'}
                  </span>
                  <button
                    onClick={() => setShowForm(false)}
                    className="text-slate-400 hover:text-white"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  <input
                    placeholder="Name *"
                    value={form.name}
                    onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <input
                    placeholder="Product name * (matches competitor)"
                    value={form.product_name}
                    onChange={(e) => setForm((f) => ({ ...f, product_name: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <input
                    placeholder="Aliases (comma-separated)"
                    value={aliasInput}
                    onChange={(e) => setAliasInput(e.target.value)}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <input
                    placeholder="Category"
                    value={form.category}
                    onChange={(e) => setForm((f) => ({ ...f, category: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <input
                    placeholder="Affiliate URL *"
                    value={form.affiliate_url}
                    onChange={(e) => setForm((f) => ({ ...f, affiliate_url: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <select
                    value={form.commission_type}
                    onChange={(e) => setForm((f) => ({ ...f, commission_type: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-cyan-500/50"
                  >
                    {COMMISSION_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                  <input
                    placeholder="Commission value (e.g. $150/signup)"
                    value={form.commission_value}
                    onChange={(e) => setForm((f) => ({ ...f, commission_value: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <input
                    placeholder="Notes"
                    value={form.notes}
                    onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
                    className="bg-slate-900/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <div className="flex items-center gap-3 pt-1">
                  <button
                    onClick={handleSave}
                    disabled={saving || !form.name || !form.product_name || !form.affiliate_url}
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

            {loading ? (
              <DataTable columns={partnerColumns} data={[]} skeletonRows={3} />
            ) : (
              <DataTable
                columns={partnerColumns}
                data={partners}
                emptyMessage="No affiliate partners registered"
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
