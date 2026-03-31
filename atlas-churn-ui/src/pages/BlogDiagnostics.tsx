import { Link } from 'react-router-dom'
import { AlertTriangle, FileSearch, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import BlogQualityDiagnosticsPanel from '../components/BlogQualityDiagnosticsPanel'
import BlogQualityTrends from '../components/BlogQualityTrends'
import useApiData from '../hooks/useApiData'
import {
  fetchBlogQualityDiagnostics,
  fetchBlogQualityTrends,
} from '../api/client'

interface BlogDiagnosticsData {
  diagnostics: Awaited<ReturnType<typeof fetchBlogQualityDiagnostics>>
  trends: Awaited<ReturnType<typeof fetchBlogQualityTrends>>
}

export default function BlogDiagnostics() {
  const {
    data,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData<BlogDiagnosticsData>(
    async () => {
      const [diagnostics, trends] = await Promise.all([
        fetchBlogQualityDiagnostics({ days: 14, top_n: 10 }),
        fetchBlogQualityTrends({ days: 14, top_n: 5 }),
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
            <h1 className="text-2xl font-bold text-white">Blog Diagnostics</h1>
            <p className="mt-1 text-sm text-slate-400">
              Inspect why blog drafts fail and whether the copy ignored available evidence or lacked upstream context.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/blog-review"
            className="inline-flex items-center gap-2 rounded-lg bg-slate-800/50 px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-700/50"
          >
            <FileSearch className="h-4 w-4" />
            Blog Review
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

      <BlogQualityDiagnosticsPanel
        data={data?.diagnostics}
        loading={loading}
      />

      <BlogQualityTrends
        data={data?.trends}
        loading={loading}
        title="Blog Failure Trendline"
      />
    </div>
  )
}
