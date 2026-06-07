import { Link } from 'react-router-dom'
import { ArrowRight, Loader2, Palette, Plus, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import {
  fetchContentOpsBrandVoiceProfiles,
  type ContentOpsBrandVoiceProfile,
} from '../api/contentOps'
import useApiData from '../hooks/useApiData'
import { PageError } from '../components/ErrorBoundary'

export default function ContentOpsBrandVoiceSettings() {
  const {
    data: profiles,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData(() => fetchContentOpsBrandVoiceProfiles(), [])

  if ((loading || refreshing) && !profiles) {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
        Loading brand voice profiles...
      </div>
    )
  }

  if (error && !profiles) {
    return <PageError error={error} onRetry={refresh} />
  }

  const savedProfiles = profiles ?? []

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header className="flex flex-col gap-4 border-b border-slate-800 pb-5 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm text-cyan-300">
            <Palette className="h-4 w-4" />
            Content Ops
          </div>
          <h1 className="text-3xl font-semibold text-white">Brand Voice</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            Saved profiles used by generated landing pages, blog posts, and
            social posts.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={refresh}
            disabled={refreshing}
            className="inline-flex items-center justify-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:bg-slate-800 disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
          <Link
            to="/content-ops/new"
            className="inline-flex items-center justify-center gap-2 rounded-md bg-cyan-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-400"
          >
            <Plus className="h-4 w-4" />
            New profile
          </Link>
        </div>
      </header>

      {error && profiles && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
          Refresh failed: {error.message}
        </div>
      )}

      {savedProfiles.length === 0 ? (
        <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-6">
          <h2 className="text-lg font-medium text-slate-100">
            No saved brand voice profiles
          </h2>
          <p className="mt-2 max-w-xl text-sm text-slate-400">
            Create one from the New Run page, then select it for generation.
          </p>
          <Link
            to="/content-ops/new"
            className="mt-4 inline-flex items-center justify-center gap-2 rounded-md border border-cyan-500/60 px-3 py-2 text-sm text-cyan-100 hover:bg-cyan-500/10"
          >
            Open New Run
            <ArrowRight className="h-4 w-4" />
          </Link>
        </section>
      ) : (
        <section className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {savedProfiles.map((profile) => (
            <BrandVoiceProfileCard key={profile.id} profile={profile} />
          ))}
        </section>
      )}
    </div>
  )
}

function BrandVoiceProfileCard({
  profile,
}: {
  profile: ContentOpsBrandVoiceProfile
}) {
  return (
    <article className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-medium text-slate-100">
            {profile.name}
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Updated {formatBrandVoiceDate(profile.updated_at)}
          </p>
        </div>
        <Link
          to="/content-ops/new"
          className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800"
        >
          Open New Run
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      <dl className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <BrandVoiceField label="Descriptors" values={profile.descriptors} />
        <BrandVoiceField label="Banned terms" values={profile.banned_terms} />
        <BrandVoiceField
          label="POV"
          values={profile.preferred_pov ? [profile.preferred_pov] : []}
        />
        <BrandVoiceField
          label="Reading"
          values={profile.reading_level ? [profile.reading_level] : []}
        />
      </dl>
      {profile.exemplars.length > 0 && (
        <div className="mt-4 rounded-md border border-slate-800 bg-slate-950/40 px-3 py-2">
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Exemplar
          </div>
          <p className="mt-1 line-clamp-3 text-sm text-slate-300">
            {profile.exemplars[0]}
          </p>
        </div>
      )}
    </article>
  )
}

function BrandVoiceField({
  label,
  values,
}: {
  label: string
  values: string[]
}) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </dt>
      <dd className="mt-1 text-slate-300">
        {values.length > 0 ? values.slice(0, 3).join(', ') : 'Not set'}
      </dd>
    </div>
  )
}

function formatBrandVoiceDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'unknown'
  }
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
