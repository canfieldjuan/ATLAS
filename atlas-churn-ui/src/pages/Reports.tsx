/* eslint-disable react-refresh/only-export-components */
import { useNavigate } from 'react-router-dom'
import { FileBarChart, RefreshCw, Search, X, Loader2 } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import UpgradeGate from '../components/UpgradeGate'
import useApiData from '../hooks/useApiData'
import { usePlanGate } from '../hooks/usePlanGate'
import { fetchReports, generateAccountComparisonReport, generateAccountDeepDiveReport, generateVendorComparisonReport } from '../api/client'
import { useState, useEffect } from 'react'
import type { Report } from '../types'

export const REPORT_TYPE_COLORS: Record<string, string> = {
  weekly_churn_feed: 'bg-cyan-500/20 text-cyan-400',
  vendor_scorecard: 'bg-violet-500/20 text-violet-400',
  displacement_report: 'bg-amber-500/20 text-amber-400',
  category_overview: 'bg-emerald-500/20 text-emerald-400',
  exploratory_overview: 'bg-slate-500/20 text-slate-300',
  vendor_comparison: 'bg-fuchsia-500/20 text-fuchsia-300',
  account_comparison: 'bg-rose-500/20 text-rose-300',
  account_deep_dive: 'bg-pink-500/20 text-pink-300',
  vendor_retention: 'bg-orange-500/20 text-orange-400',
  challenger_intel: 'bg-purple-500/20 text-purple-400',
}

function CardSkeleton() {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 animate-pulse">
      <div className="flex items-start justify-between mb-3">
        <div className="h-5 w-28 bg-slate-700/50 rounded" />
        <div className="h-4 w-4 bg-slate-700/50 rounded" />
      </div>
      <div className="h-4 w-36 bg-slate-700/50 rounded mb-2" />
      <div className="h-3 w-full bg-slate-700/50 rounded mb-1" />
      <div className="h-3 w-3/4 bg-slate-700/50 rounded" />
      <div className="h-3 w-20 bg-slate-700/50 rounded mt-4" />
    </div>
  )
}

export default function Reports() {
  const navigate = useNavigate()
  const { canAccessReports } = usePlanGate()
  const [typeFilter, setTypeFilter] = useState('')
  const [vendorSearch, setVendorSearch] = useState('')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [primaryVendor, setPrimaryVendor] = useState('')
  const [comparisonVendor, setComparisonVendor] = useState('')
  const [primaryCompany, setPrimaryCompany] = useState('')
  const [comparisonCompany, setComparisonCompany] = useState('')
  const [deepDiveCompany, setDeepDiveCompany] = useState('')
  const [creatingComparison, setCreatingComparison] = useState(false)
  const [creatingAccountComparison, setCreatingAccountComparison] = useState(false)
  const [creatingAccountDeepDive, setCreatingAccountDeepDive] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(timer)
  }, [vendorSearch])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchReports({ report_type: typeFilter || undefined, vendor_filter: debouncedVendor || undefined, limit: 50 }),
    [typeFilter, debouncedVendor],
  )

  const reports = data?.reports ?? []
  const hasFilters = typeFilter !== '' || vendorSearch !== ''
  const debouncePending = vendorSearch !== debouncedVendor

  function clearFilters() {
    setTypeFilter('')
    setVendorSearch('')
    setDebouncedVendor('')
  }

  async function handleCreateComparison() {
    if (!primaryVendor.trim() || !comparisonVendor.trim()) {
      alert('Enter both vendors to compare')
      return
    }
    setCreatingComparison(true)
    try {
      const result = await generateVendorComparisonReport({
        primary_vendor: primaryVendor.trim(),
        comparison_vendor: comparisonVendor.trim(),
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setPrimaryVendor('')
      setComparisonVendor('')
      refresh()
      if (reportId) {
        navigate(`/reports/${reportId}`)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Comparison generation failed')
    } finally {
      setCreatingComparison(false)
    }
  }

  async function handleCreateAccountComparison() {
    if (!primaryCompany.trim() || !comparisonCompany.trim()) {
      alert('Enter both companies to compare')
      return
    }
    setCreatingAccountComparison(true)
    try {
      const result = await generateAccountComparisonReport({
        primary_company: primaryCompany.trim(),
        comparison_company: comparisonCompany.trim(),
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setPrimaryCompany('')
      setComparisonCompany('')
      refresh()
      if (reportId) {
        navigate(`/reports/${reportId}`)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Account comparison generation failed')
    } finally {
      setCreatingAccountComparison(false)
    }
  }

  async function handleCreateAccountDeepDive() {
    if (!deepDiveCompany.trim()) {
      alert('Enter a company for the account deep dive')
      return
    }
    setCreatingAccountDeepDive(true)
    try {
      const result = await generateAccountDeepDiveReport({
        company_name: deepDiveCompany.trim(),
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setDeepDiveCompany('')
      refresh()
      if (reportId) {
        navigate(`/reports/${reportId}`)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Account deep dive generation failed')
    } finally {
      setCreatingAccountDeepDive(false)
    }
  }

  if (!canAccessReports) {
    return (
      <UpgradeGate allowed={false} feature="Intelligence Reports" requiredPlan="Starter">
        <div />
      </UpgradeGate>
    )
  }

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">Intelligence Reports</h1>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 flex-wrap">
          <div className="relative flex-1 max-w-xs w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              type="text"
              placeholder="Filter by vendor..."
              value={vendorSearch}
              onChange={(e) => setVendorSearch(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
          >
            <option value="">All Types</option>
            <option value="weekly_churn_feed">Weekly Churn Feed</option>
            <option value="vendor_scorecard">Vendor Scorecard</option>
            <option value="displacement_report">Displacement Report</option>
            <option value="category_overview">Category Overview</option>
            <option value="exploratory_overview">Exploratory Overview</option>
            <option value="vendor_comparison">Vendor Comparison</option>
            <option value="account_comparison">Account Comparison</option>
            <option value="account_deep_dive">Account Deep Dive</option>
            <option value="vendor_retention">Vendor Retention</option>
            <option value="challenger_intel">Challenger Intel</option>
          </select>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
            >
              <X className="h-3 w-3" />
              Clear filters
            </button>
          )}
        </div>

        <div className="flex items-center gap-2 text-sm text-slate-400">
          {debouncePending || loading ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Searching...
            </>
          ) : (
            <span>{reports.length} report{reports.length !== 1 ? 's' : ''} found</span>
          )}
        </div>
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-400 mb-1">Primary vendor</label>
            <input
              value={primaryVendor}
              onChange={(e) => setPrimaryVendor(e.target.value)}
              placeholder="Example: Salesforce"
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-400 mb-1">Comparison vendor</label>
            <input
              value={comparisonVendor}
              onChange={(e) => setComparisonVendor(e.target.value)}
              placeholder="Example: HubSpot"
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <button
            onClick={handleCreateComparison}
            disabled={creatingComparison}
            className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-fuchsia-500/15 text-fuchsia-300 text-sm font-medium hover:bg-fuchsia-500/25 transition-colors disabled:opacity-50"
          >
            {creatingComparison ? 'Generating...' : 'Create Comparison'}
          </button>
        </div>
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-400 mb-1">Account deep dive company</label>
            <input
              value={deepDiveCompany}
              onChange={(e) => setDeepDiveCompany(e.target.value)}
              placeholder="Example: DataPulse Analytics"
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <button
            onClick={handleCreateAccountDeepDive}
            disabled={creatingAccountDeepDive}
            className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-pink-500/15 text-pink-300 text-sm font-medium hover:bg-pink-500/25 transition-colors disabled:opacity-50"
          >
            {creatingAccountDeepDive ? 'Generating...' : 'Create Account Deep Dive'}
          </button>
        </div>
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-400 mb-1">Primary company</label>
            <input
              value={primaryCompany}
              onChange={(e) => setPrimaryCompany(e.target.value)}
              placeholder="Example: DataPulse Analytics"
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-400 mb-1">Comparison company</label>
            <input
              value={comparisonCompany}
              onChange={(e) => setComparisonCompany(e.target.value)}
              placeholder="Example: FinEdge Capital"
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <button
            onClick={handleCreateAccountComparison}
            disabled={creatingAccountComparison}
            className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-rose-500/15 text-rose-300 text-sm font-medium hover:bg-rose-500/25 transition-colors disabled:opacity-50"
          >
            {creatingAccountComparison ? 'Generating...' : 'Create Account Comparison'}
          </button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }, (_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : reports.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <FileBarChart className="h-10 w-10 text-slate-600 mb-4" />
          <p className="text-slate-500 mb-4">No reports found</p>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-medium hover:bg-cyan-500/20 transition-colors"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reports.map((r: Report) => (
            <button
              key={r.id}
              onClick={() => navigate(`/reports/${r.id}`)}
              className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5 text-left hover:border-cyan-500/30 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    REPORT_TYPE_COLORS[r.report_type] ?? 'bg-slate-500/20 text-slate-400'
                  }`}
                >
                  {r.report_type.replace(/_/g, ' ')}
                </span>
                <FileBarChart className="h-4 w-4 text-slate-500" />
              </div>
              {(r.vendor_filter || r.category_filter) && (
                <p className="text-sm text-white font-medium mb-1">
                  {['vendor_comparison', 'account_comparison'].includes(r.report_type) && r.vendor_filter && r.category_filter
                    ? `${r.vendor_filter} vs ${r.category_filter}`
                    : r.vendor_filter}
                </p>
              )}
              <p className="text-sm text-slate-400 line-clamp-2">
                {r.executive_summary ?? 'No summary available'}
              </p>
              <p className="text-xs text-slate-500 mt-3">
                {r.report_date ?? r.created_at ?? '--'}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
