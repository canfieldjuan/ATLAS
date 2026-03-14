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
  model: string
  provider: string
  input_tokens: number
  output_tokens: number
  cost_usd: number
  duration_ms: number
  tokens_per_second: number
  status: string
  created_at: string
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
  reviews_found: number
  reviews_inserted: number
  insert_rate: number
  avg_duration_ms: number
  captcha_attempts: number
}

export interface ScrapeQuality {
  source: string
  total_reviews: number
  high_signal_reviews: number
  high_signal_rate: number
  enriched_reviews: number
  enrichment_rate: number
  avg_source_weight: number
  high_value_authors: number
}

export interface ScrapeDetail {
  id: string
  source: string
  status: string
  vendor_name: string
  product_name: string
  reviews_found: number
  reviews_inserted: number
  insert_rate: number
  duration_ms: number
  error_count: number
  errors: string[]
  captcha_attempts: number
  block_type: string | null
  parser_version: string
  proxy_type: string
  started_at: string
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
