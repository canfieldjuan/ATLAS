import type {
  AdminDashboardData,
  DailyEntry,
  ErrorTimelineEntry,
  ModelEntry,
  ProviderEntry,
  ReasoningActivityData,
  RecentCall,
  RedditBySubredditData,
  RedditOverview,
  RedditPerVendorData,
  RedditSignalBreakdownData,
  ScrapeDetail,
  ScrapeRunPagesData,
  ScrapeSummaryData,
  ScrapeTopPost,
  Summary,
  SystemResources,
  TaskHealth,
  WorkflowEntry,
} from '../types'

const ADMIN_COSTS_BASE = '/api/v1/admin/costs'

type ProviderResponse = { providers?: ProviderEntry[] }
type ModelResponse = { models?: ModelEntry[] }
type WorkflowResponse = { workflows?: WorkflowEntry[] }
type DailyResponse = { daily?: DailyEntry[] }
type RecentCallsResponse = { calls?: RecentCall[] }
type TaskHealthResponse = { tasks?: TaskHealth[] }
type ErrorTimelineResponse = { daily?: ErrorTimelineEntry[] }
type ScrapeSummaryResponse = ScrapeSummaryData
type ScrapeDetailsResponse = { scrapes?: ScrapeDetail[] }
type ScrapeTopPostsResponse = { posts?: ScrapeTopPost[] }

function normalizeTextLabel(value: string | null | undefined, fallback: string): string {
  const text = (value ?? '').trim()
  return text ? text : fallback
}

function dedupeLabels(values: string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    const label = normalizeTextLabel(value, 'unknown')
    const key = label.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    result.push(label)
  }
  return result
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${ADMIN_COSTS_BASE}${path}`)
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

async function fetchOptionalJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(`${ADMIN_COSTS_BASE}${path}`)
    if (!response.ok) return fallback
    return response.json() as Promise<T>
  } catch {
    return fallback
  }
}

function normalizeProviders(providers: ProviderEntry[] | undefined): ProviderEntry[] {
  return (providers ?? []).map((provider) => ({
    ...provider,
    total_tokens: provider.total_tokens || (provider.input_tokens + provider.output_tokens),
  }))
}

function normalizeRedditBySubreddit(data: RedditBySubredditData | null): RedditBySubredditData | null {
  if (!data) return null
  const grouped = new Map<string, {
    entry: RedditBySubredditData['subreddits'][number]
    weightSum: number
    urgencySum: number
    scoreSum: number
  }>()

  for (const entry of data.subreddits) {
    const subreddit = normalizeTextLabel(entry.subreddit, 'unknown')
    const key = subreddit.toLowerCase()
    const existing = grouped.get(key)
    if (!existing) {
      grouped.set(key, {
        entry: { ...entry, subreddit },
        weightSum: entry.avg_source_weight * entry.total_posts,
        urgencySum: entry.avg_urgency_score * entry.enriched_posts,
        scoreSum: entry.avg_reddit_score * entry.total_posts,
      })
      continue
    }

    const merged = existing.entry
    const totalPosts = merged.total_posts + entry.total_posts
    const triageBase = merged.enriched_posts + merged.no_signal_posts + entry.enriched_posts + entry.no_signal_posts
    grouped.set(key, {
      entry: {
        subreddit: merged.subreddit,
        total_posts: totalPosts,
        high_signal_posts: merged.high_signal_posts + entry.high_signal_posts,
        signal_rate: totalPosts > 0 ? (merged.high_signal_posts + entry.high_signal_posts) / totalPosts : 0,
        enriched_posts: merged.enriched_posts + entry.enriched_posts,
        triage_pass_rate: triageBase > 0 ? (merged.enriched_posts + entry.enriched_posts) / triageBase : 0,
        no_signal_posts: merged.no_signal_posts + entry.no_signal_posts,
        failed_posts: merged.failed_posts + entry.failed_posts,
        avg_source_weight: 0,
        avg_urgency_score: 0,
        avg_reddit_score: 0,
        trending_high_count: merged.trending_high_count + entry.trending_high_count,
        comment_harvested_count: merged.comment_harvested_count + entry.comment_harvested_count,
      },
      weightSum: existing.weightSum + (entry.avg_source_weight * entry.total_posts),
      urgencySum: existing.urgencySum + (entry.avg_urgency_score * entry.enriched_posts),
      scoreSum: existing.scoreSum + (entry.avg_reddit_score * entry.total_posts),
    })
  }

  return {
    ...data,
    subreddits: Array.from(grouped.values()).map(({ entry, weightSum, urgencySum, scoreSum }) => ({
      ...entry,
      avg_source_weight: entry.total_posts > 0 ? weightSum / entry.total_posts : 0,
      avg_urgency_score: entry.enriched_posts > 0 ? urgencySum / entry.enriched_posts : 0,
      avg_reddit_score: entry.total_posts > 0 ? scoreSum / entry.total_posts : 0,
    })).sort((a, b) => b.enriched_posts - a.enriched_posts || b.total_posts - a.total_posts),
  }
}

function normalizeRedditSignalBreakdown(data: RedditSignalBreakdownData | null): RedditSignalBreakdownData | null {
  if (!data) return null
  const grouped = new Map<string, RedditSignalBreakdownData['flair_analysis'][number]>()

  for (const entry of data.flair_analysis) {
    const flair = normalizeTextLabel(entry.flair, '(no flair)')
    const key = flair.toLowerCase()
    const existing = grouped.get(key)
    if (!existing) {
      grouped.set(key, { ...entry, flair })
      continue
    }
    const totalCount = existing.count + entry.count
    grouped.set(key, {
      flair: existing.flair,
      count: totalCount,
      signal_rate: totalCount > 0 ? ((existing.signal_rate * existing.count) + (entry.signal_rate * entry.count)) / totalCount : 0,
      avg_weight: totalCount > 0 ? ((existing.avg_weight * existing.count) + (entry.avg_weight * entry.count)) / totalCount : 0,
    })
  }

  return {
    ...data,
    flair_analysis: Array.from(grouped.values()).sort((a, b) => b.count - a.count),
  }
}

function normalizeRedditPerVendor(data: RedditPerVendorData | null): RedditPerVendorData | null {
  if (!data) return null
  return {
    ...data,
    vendors: data.vendors.map((vendor) => ({
      ...vendor,
      top_subreddits: dedupeLabels(vendor.top_subreddits),
      top_pain_categories: dedupeLabels(vendor.top_pain_categories),
    })),
  }
}

export async function fetchAdminDashboardData(days: number): Promise<AdminDashboardData> {
  const [
    summary,
    providerData,
    modelData,
    workflowData,
    dailyData,
    recentData,
    taskData,
    errorData,
    sysResources,
    scrapeSummary,
    scrapeDetails,
    scrapeTopPosts,
    reasoningActivity,
    redditOverview,
    redditBySubreddit,
    redditSignalBreakdown,
    redditPerVendor,
  ] = await Promise.all([
    fetchJson<Summary>(`/summary?days=${days}`),
    fetchJson<ProviderResponse>(`/by-provider?days=${days}`),
    fetchOptionalJson<ModelResponse>(`/by-model?days=${days}`, {}),
    fetchJson<WorkflowResponse>(`/by-workflow?days=${days}`),
    fetchJson<DailyResponse>(`/daily?days=${days}`),
    fetchJson<RecentCallsResponse>('/recent?limit=50'),
    fetchOptionalJson<TaskHealthResponse>(`/task-health?days=${days}`, {}),
    fetchOptionalJson<ErrorTimelineResponse>(`/error-timeline?days=${days}`, {}),
    fetchOptionalJson<SystemResources | null>('/system-resources', null),
    fetchOptionalJson<ScrapeSummaryResponse | null>(`/scraping/summary?days=${days}`, null),
    fetchOptionalJson<ScrapeDetailsResponse>('/scraping/details?limit=50', {}),
    fetchOptionalJson<ScrapeTopPostsResponse>('/scraping/top-posts?source=reddit&limit=25', {}),
    fetchOptionalJson<ReasoningActivityData | null>(`/reasoning-activity?days=${days}`, null),
    fetchOptionalJson<RedditOverview | null>(`/scraping/reddit/overview?days=${days}`, null),
    fetchOptionalJson<RedditBySubredditData | null>(`/scraping/reddit/by-subreddit?days=${days}`, null),
    fetchOptionalJson<RedditSignalBreakdownData | null>(`/scraping/reddit/signal-breakdown?days=${days}`, null),
    fetchOptionalJson<RedditPerVendorData | null>(`/scraping/reddit/per-vendor?days=${days}`, null),
  ])

  return {
    summary,
    providers: normalizeProviders(providerData.providers),
    models: modelData.models ?? [],
    workflows: workflowData.workflows ?? [],
    daily: dailyData.daily ?? [],
    recent: recentData.calls ?? [],
    tasks: taskData.tasks ?? [],
    errorTimeline: errorData.daily ?? [],
    sysResources,
    scrapeSummary,
    scrapeDetails: scrapeDetails.scrapes ?? [],
    scrapeTopPosts: scrapeTopPosts.posts ?? [],
    reasoningActivity,
    redditOverview,
    redditBySubreddit: normalizeRedditBySubreddit(redditBySubreddit),
    redditSignalBreakdown: normalizeRedditSignalBreakdown(redditSignalBreakdown),
    redditPerVendor: normalizeRedditPerVendor(redditPerVendor),
  }
}

export async function fetchSystemResources(): Promise<SystemResources | null> {
  return fetchOptionalJson<SystemResources | null>('/system-resources', null)
}

export async function fetchScrapeRunPages(runId: string): Promise<ScrapeRunPagesData> {
  return fetchJson<ScrapeRunPagesData>(`/scraping/runs/${runId}/pages`)
}
