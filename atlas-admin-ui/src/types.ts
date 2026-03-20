export interface Summary {
  total_cost_usd: number
  today_cost_usd: number
  total_calls: number
  today_calls: number
  avg_tokens_per_second: number
  total_tokens: number
  total_input_tokens: number
  total_output_tokens: number
  avg_duration_ms: number
}

export interface ProviderEntry {
  provider: string
  cost_usd: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  calls: number
}

export interface ModelEntry {
  model: string
  provider: string
  cost_usd: number
  total_tokens: number
  calls: number
  avg_duration_ms: number
  avg_tokens_per_second: number
}

export interface WorkflowEntry {
  workflow: string
  cost_usd: number
  total_tokens: number
  calls: number
  avg_duration_ms: number
}

export interface DailyEntry {
  date: string
  cost_usd: number
  total_tokens: number
  calls: number
}

export interface MergedDailyEntry extends DailyEntry {
  error_calls: number
  error_rate: number
}

export interface RecentCall {
  span_name: string
  title?: string
  detail?: string | null
  model: string
  provider: string
  input_tokens: number
  output_tokens: number
  cost_usd: number
  duration_ms: number
  tokens_per_second: number
  status: string
  metadata?: Record<string, unknown>
  created_at: string | null
}

export interface TaskHealth {
  id: string
  name: string
  task_type: string
  schedule_type: string
  cron_expression: string | null
  interval_seconds: number | null
  enabled: boolean
  last_run_at: string | null
  next_run_at: string | null
  last_status: string | null
  last_duration_ms: number | null
  last_error: string | null
  recent_failure_rate: number
  recent_runs: number
}

export interface ErrorTimelineEntry {
  date: string
  total_calls: number
  error_calls: number
  error_rate: number
}

export interface ReasoningPhase {
  span_name: string
  pass_type: string
  pass_number: number
  calls: number
  cost_usd: number
  total_tokens: number
  avg_duration_ms: number
  changed_count: number
}

export interface ReasoningActivitySummary {
  total_cost_usd: number
  total_tokens: number
  total_calls: number
}

export interface ReasoningActivityData {
  period_days: number
  phases: ReasoningPhase[]
  summary: ReasoningActivitySummary
}

export interface SystemResources {
  cpu_percent: number
  mem_percent: number
  mem_used_gb: number
  mem_total_gb: number
  net_mbps: number
  gpu: {
    name: string
    utilization_percent: number
    vram_used_gb: number
    vram_total_gb: number
    vram_percent: number
    temperature_c: number
  } | null
}

export interface ScrapeThroughput {
  source: string
  vendor_name: string
  total_runs: number
  successes: number
  failures: number
  blocked: number
  partial: number
  reviews_found: number
  reviews_inserted: number
  insert_rate: number
  avg_duration_ms: number
  captcha_attempts: number
  blocked_requests: number
}

export interface ScrapeQuality {
  source: string
  total_reviews: number
  high_signal_reviews: number
  high_signal_rate: number
  enriched_reviews: number
  enrichment_rate: number
  failed_enrichments: number
  avg_source_weight: number
  high_value_authors: number
}

export interface ScrapeDetail {
  id: string
  source: string
  status: string
  vendor_name: string
  product_name: string
  product_slug: string
  reviews_found: number
  reviews_inserted: number
  insert_rate: number
  pages_scraped: number
  duration_ms: number
  error_count: number
  errors: string[]
  captcha_attempts: number
  captcha_types: string[]
  captcha_solve_ms: number | null
  block_type: string | null
  parser_version: string
  proxy_type: string
  stop_reason: string | null
  oldest_review: string | null
  newest_review: string | null
  date_dropped: number
  duplicate_pages: number
  has_page_logs: boolean
  started_at: string | null
}

export interface ScrapeRunPage {
  page: number
  url: string
  requested_at: string | null
  status_code: number | null
  final_url: string | null
  response_bytes: number | null
  duration_ms: number | null
  review_nodes_found: number | null
  reviews_parsed: number | null
  missing_date: number | null
  missing_rating: number | null
  missing_body: number | null
  missing_author: number | null
  oldest_review: string | null
  newest_review: string | null
  next_page_found: boolean | null
  next_page_url: string | null
  content_hash: string | null
  duplicate_reviews: number | null
  stop_reason: string | null
  errors: string[]
}

export interface ScrapeRunPagesData {
  run: {
    id: string
    source: string
    vendor_name: string
    status: string
    stop_reason: string | null
    pages_scraped: number
    reviews_found: number
    reviews_inserted: number
    oldest_review: string | null
    newest_review: string | null
    date_dropped: number
    duplicate_pages: number
    duration_ms: number | null
    started_at: string | null
  }
  pages: ScrapeRunPage[]
  page_count: number
}

export interface ScrapeTopPost {
  id: string
  vendor_name: string
  summary: string
  source_url: string
  reviewed_at: string | null
  imported_at: string | null
  enrichment_status: string
  source_weight: number
  trending_score: string
  author_churn_score: number
  subreddit: string
  reddit_score: number
  num_comments: number
  post_flair: string
  is_edited: boolean
  is_crosspost: boolean
  comment_count: number
}

export interface ScrapeSummaryData {
  today: { runs: number; reviews_inserted: number; errors: number }
  throughput: ScrapeThroughput[]
  quality: ScrapeQuality[]
}

export interface RedditOverview {
  period_days: number
  auth_mode: string
  runs: {
    total: number
    completed: number
    failed: number
    blocked: number
    partial: number
  }
  throughput: {
    reviews_found: number
    reviews_inserted: number
    insert_rate: number
    avg_duration_ms: number
    pages_scraped_total: number
  }
  rate_limits: {
    runs_with_429s: number
    total_429_events: number
  }
  signal_funnel: {
    inserted: number
    enriched: number
    no_signal: number
    failed: number
    pending: number
    triage_pass_rate: number
    enrichment_completion_rate: number
  }
  signal_conversion: {
    intent_to_leave: number
    high_urgency: number
    actionable: number
    actionable_rate: number
  }
}

export interface RedditSubredditEntry {
  subreddit: string
  total_posts: number
  high_signal_posts: number
  signal_rate: number
  enriched_posts: number
  triage_pass_rate: number
  no_signal_posts: number
  failed_posts: number
  avg_source_weight: number
  avg_urgency_score: number
  avg_reddit_score: number
  trending_high_count: number
  comment_harvested_count: number
}

export interface RedditBySubredditData {
  period_days: number
  subreddits: RedditSubredditEntry[]
}

export interface RedditSignalBreakdownData {
  period_days: number
  flair_analysis: Array<{
    flair: string
    count: number
    signal_rate: number
    avg_weight: number
  }>
  edit_stats: {
    edited_posts: number
    edited_signal_rate: number
    unedited_signal_rate: number
  }
  crosspost_stats: {
    crossposts: number
    crosspost_signal_rate: number
    crosspost_subreddits_reached: number
  }
  comment_harvest_stats: {
    posts_with_comments: number
    avg_comments_fetched: number
    comment_trigger_rate: number
  }
  author_churn_stats: {
    high_score_authors: number
    avg_churn_score: number
    score_distribution: Record<string, number>
  }
  post_age_distribution: Record<string, number>
  trending_distribution: Record<string, number>
}

export interface RedditVendorEntry {
  vendor_name: string
  inserted: number
  enriched: number
  no_signal: number
  failed: number
  triage_pass_rate: number
  avg_source_weight: number
  avg_urgency_score: number
  intent_to_leave_count: number
  high_urgency_count: number
  trending_high_count: number
  top_subreddits: string[]
  top_pain_categories: string[]
}

export interface RedditPerVendorData {
  period_days: number
  vendors: RedditVendorEntry[]
}

export interface AdminDashboardData {
  summary: Summary
  providers: ProviderEntry[]
  models: ModelEntry[]
  workflows: WorkflowEntry[]
  daily: DailyEntry[]
  recent: RecentCall[]
  tasks: TaskHealth[]
  errorTimeline: ErrorTimelineEntry[]
  sysResources: SystemResources | null
  scrapeSummary: ScrapeSummaryData | null
  scrapeDetails: ScrapeDetail[]
  scrapeTopPosts: ScrapeTopPost[]
  reasoningActivity: ReasoningActivityData | null
  redditOverview: RedditOverview | null
  redditBySubreddit: RedditBySubredditData | null
  redditSignalBreakdown: RedditSignalBreakdownData | null
  redditPerVendor: RedditPerVendorData | null
}
