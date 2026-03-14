import type { TaskHealth } from './types'

export const PROVIDER_COLORS: Record<string, string> = {
  anthropic: '#f97316',
  groq: '#10b981',
  openrouter: '#8b5cf6',
  ollama: '#06b6d4',
  vllm: '#3b82f6',
  deepseek: '#eab308',
  local: '#6b7280',
  unknown: '#475569',
}

export function fmtCost(v: number): string {
  if (v >= 100) return `$${v.toFixed(0)}`
  if (v >= 1) return `$${v.toFixed(2)}`
  if (v >= 0.01) return `$${v.toFixed(3)}`
  return `$${v.toFixed(4)}`
}

export function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

export function fmtDuration(ms: number): string {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

export function timeAgo(iso: string): string {
  const ts = new Date(iso).getTime()
  if (Number.isNaN(ts)) return ''
  const diff = Date.now() - ts
  if (diff < 0) return 'just now'
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

export function provColor(p: string): string {
  return PROVIDER_COLORS[p?.toLowerCase()] || PROVIDER_COLORS.unknown
}

export function fmtDate(v: string): string {
  return new Date(v + 'T00:00').toLocaleDateString('en', { month: 'short', day: 'numeric' })
}

export function gaugeColor(pct: number, thresholds: [number, number] = [50, 80]): string {
  if (pct < thresholds[0]) return 'bg-emerald-400'
  if (pct < thresholds[1]) return 'bg-amber-400'
  return 'bg-red-400'
}

export function gaugeTextColor(pct: number, thresholds: [number, number] = [50, 80]): string {
  if (pct < thresholds[0]) return 'text-emerald-400'
  if (pct < thresholds[1]) return 'text-amber-400'
  return 'text-red-400'
}

export function isFailing(t: TaskHealth): boolean {
  return t.enabled && (t.last_status === 'failed' || t.last_status === 'timeout' || t.last_status === 'error')
}

export function fmtSchedule(task: TaskHealth): string {
  if (task.cron_expression) return task.cron_expression
  if (task.interval_seconds) {
    const s = task.interval_seconds
    if (s >= 3600) return `every ${(s / 3600).toFixed(0)}h`
    if (s >= 60) return `every ${(s / 60).toFixed(0)}m`
    return `every ${s}s`
  }
  return task.schedule_type
}
