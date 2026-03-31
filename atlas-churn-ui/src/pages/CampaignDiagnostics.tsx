import { Link } from 'react-router-dom'
import { AlertTriangle, MailSearch, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import CampaignQualityDiagnosticsPanel from '../components/CampaignQualityDiagnosticsPanel'
import CampaignQualityTrends from '../components/CampaignQualityTrends'
import useApiData from '../hooks/useApiData'
import {
  fetchCampaignQualityDiagnostics,
  fetchCampaignQualityTrends,
} from '../api/client'

interface CampaignDiagnosticsData {
  diagnostics: Awaited<ReturnType<typeof fetchCampaignQualityDiagnostics>>
  trends: Awaited<ReturnType<typeof fetchCampaignQualityTrends>>
}

export default function CampaignDiagnostics() {
  const {
    data,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData<CampaignDiagnosticsData>(
    async () => {
      const [diagnostics, trends] = await Promise.all([
        fetchCampaignQualityDiagnostics({ days: 14, top_n: 10 }),
        fetchCampaignQualityTrends({ days: 14, top_n: 5 }),
      ])
      return { diagnostics, trends }
    },
    [],
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex items-center gap-3">
          <AlertTriangle className="h-6 w-6 text-red-300" />
          <div>
            <h1 className="text-2xl font-bold text-white">Campaign Diagnostics</h1>
            <p className="mt-1 text-sm text-slate-400">
              Inspect why campaign failures occurred and what evidence was missing or ignored.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/campaign-review"
            className="inline-flex items-center gap-2 rounded-lg bg-slate-800/50 px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-700/50"
          >
            <MailSearch className="h-4 w-4" />
            Review Queue
          </Link>
          <button
            onClick={() => refresh()}
            disabled={refreshing}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-800/50 px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-700/50 disabled:opacity-60"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {error.message}
        </div>
      ) : null}

      <CampaignQualityDiagnosticsPanel
        data={data?.diagnostics}
        loading={loading}
      />

      <CampaignQualityTrends
        data={data?.trends}
        loading={loading}
        title="Campaign Failure Trendline"
      />
    </div>
  )
}
