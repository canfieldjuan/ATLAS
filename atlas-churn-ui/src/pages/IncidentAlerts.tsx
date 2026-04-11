import { useMemo, useState, type FormEvent } from 'react'
import { BellRing, CheckCircle2, ChevronDown, ChevronRight, FlaskConical, RefreshCw, ShieldAlert, Trash2 } from 'lucide-react'
import StatCard from '../components/StatCard'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  createWebhook,
  deleteWebhookSubscription,
  fetchWebhookDeliverySummary,
  listWebhookCrmPushLog,
  listWebhookDeliveries,
  listWebhooks,
  testWebhookSubscription,
  updateWebhookSubscription,
  type WebhookChannel,
  type WebhookCreateBody,
  type WebhookEventType,
} from '../api/client'

const EVENT_TYPE_OPTIONS: Array<{ value: WebhookEventType; label: string; description: string }> = [
  { value: 'churn_alert', label: 'Churn Alert', description: 'High-severity incident or displacement alerts.' },
  { value: 'change_event', label: 'Change Event', description: 'Material changes in vendor or market signals.' },
  { value: 'signal_update', label: 'Signal Update', description: 'Ongoing evidence refreshes for monitored vendors.' },
  { value: 'report_generated', label: 'Report Generated', description: 'New persisted battle cards or reports.' },
]

const CHANNEL_OPTIONS: Array<{ value: WebhookChannel; label: string }> = [
  { value: 'generic', label: 'Generic JSON' },
  { value: 'slack', label: 'Slack' },
  { value: 'teams', label: 'Microsoft Teams' },
  { value: 'crm_hubspot', label: 'HubSpot CRM' },
  { value: 'crm_salesforce', label: 'Salesforce CRM' },
  { value: 'crm_pipedrive', label: 'Pipedrive CRM' },
]

const SUMMARY_WINDOWS = [7, 30, 90] as const

function formatTs(value: string | null | undefined) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleString()
}

function formatPercent(value: number | null | undefined) {
  if (value == null) return '--'
  return `${Math.round(value * 100)}%`
}

function formatDurationMs(value: number | null | undefined) {
  if (value == null) return '--'
  return `${Math.round(value)} ms`
}

function generateWebhookSecret() {
  return `atlas_${Math.random().toString(36).slice(2, 14)}${Math.random().toString(36).slice(2, 14)}`
}

function channelTone(channel: WebhookChannel) {
  if (channel === 'slack' || channel === 'teams') return 'border-violet-500/30 bg-violet-500/10 text-violet-200'
  if (channel.startsWith('crm_')) return 'border-amber-500/30 bg-amber-500/10 text-amber-200'
  return 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200'
}

function deliveryTone(success: boolean) {
  return success
    ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-100'
    : 'border-rose-500/20 bg-rose-500/5 text-rose-100'
}

export default function IncidentAlerts() {
  const [summaryWindow, setSummaryWindow] = useState<(typeof SUMMARY_WINDOWS)[number]>(7)
  const [form, setForm] = useState<WebhookCreateBody>({
    url: '',
    secret: generateWebhookSecret(),
    event_types: ['churn_alert'],
    channel: 'generic',
    auth_header: '',
    description: '',
  })
  const [saving, setSaving] = useState(false)
  const [busyWebhookId, setBusyWebhookId] = useState<string | null>(null)
  const [selectedWebhookId, setSelectedWebhookId] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const {
    data: summary,
    loading: summaryLoading,
    error: summaryError,
    refresh: refreshSummary,
    refreshing: summaryRefreshing,
  } = useApiData(
    () => fetchWebhookDeliverySummary(summaryWindow),
    [summaryWindow],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )

  const {
    data: listData,
    loading: listLoading,
    error: listError,
    refresh: refreshWebhooks,
    refreshing: listRefreshing,
  } = useApiData(
    () => listWebhooks(),
    [],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )

  const webhooks = listData?.webhooks ?? []
  const refreshing = summaryRefreshing || listRefreshing
  const loading = summaryLoading || listLoading
  const error = summaryError || listError
  const selectedWebhook = useMemo(
    () => webhooks.find((webhook) => webhook.id === selectedWebhookId) ?? null,
    [selectedWebhookId, webhooks],
  )

  const {
    data: deliveryData,
    loading: deliveryLoading,
    refresh: refreshDeliveries,
    refreshing: deliveryRefreshing,
  } = useApiData(
    () => (selectedWebhookId
      ? listWebhookDeliveries(selectedWebhookId, { limit: 10 })
      : Promise.resolve({ deliveries: [], count: 0 })),
    [selectedWebhookId],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )

  const {
    data: crmPushData,
    loading: crmPushLoading,
    refresh: refreshCrmPushLog,
    refreshing: crmPushRefreshing,
  } = useApiData(
    () => (selectedWebhookId && selectedWebhook?.channel.startsWith('crm_')
      ? listWebhookCrmPushLog(selectedWebhookId, 10)
      : Promise.resolve({ pushes: [], count: 0 })),
    [selectedWebhook?.channel, selectedWebhookId],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )

  const selectedEventCount = form.event_types.length
  const requiresAuthHeader = form.channel.startsWith('crm_')
  const sortedEvents = useMemo(() => new Set(form.event_types), [form.event_types])
  const activityLoading = deliveryLoading || crmPushLoading
  const activityRefreshing = deliveryRefreshing || crmPushRefreshing

  async function refreshAll() {
    refreshSummary()
    refreshWebhooks()
    if (selectedWebhookId) {
      refreshDeliveries()
      refreshCrmPushLog()
    }
  }

  function setFormValue<K extends keyof WebhookCreateBody>(key: K, value: WebhookCreateBody[K]) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  function toggleEventType(eventType: WebhookEventType) {
    setForm((current) => {
      const next = new Set(current.event_types)
      if (next.has(eventType)) {
        next.delete(eventType)
      } else {
        next.add(eventType)
      }
      return { ...current, event_types: Array.from(next) as WebhookEventType[] }
    })
  }

  async function handleCreateWebhook(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form.url.trim()) {
      setActionError('Webhook URL is required')
      setMessage(null)
      return
    }
    if (!form.secret.trim()) {
      setActionError('Webhook secret is required')
      setMessage(null)
      return
    }
    if (form.event_types.length === 0) {
      setActionError('Select at least one event type')
      setMessage(null)
      return
    }
    if (requiresAuthHeader && !form.auth_header?.trim()) {
      setActionError('CRM channels require an auth header')
      setMessage(null)
      return
    }
    setSaving(true)
    setActionError(null)
    setMessage(null)
    try {
      const created = await createWebhook({
        url: form.url.trim(),
        secret: form.secret.trim(),
        event_types: form.event_types,
        channel: form.channel,
        auth_header: form.auth_header?.trim() || undefined,
        description: form.description?.trim() || undefined,
      })
      setMessage(`Added ${created.channel} webhook`)
      setForm({
        url: '',
        secret: generateWebhookSecret(),
        event_types: ['churn_alert'],
        channel: 'generic',
        auth_header: '',
        description: '',
      })
      await refreshAll()
    } catch (err) {
      setMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to add webhook')
    } finally {
      setSaving(false)
    }
  }

  async function handleToggleWebhook(webhookId: string, enabled: boolean) {
    setBusyWebhookId(webhookId)
    setActionError(null)
    setMessage(null)
    try {
      await updateWebhookSubscription(webhookId, { enabled: !enabled })
      setMessage(enabled ? 'Webhook disabled' : 'Webhook enabled')
      await refreshAll()
    } catch (err) {
      setMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to update webhook')
    } finally {
      setBusyWebhookId(null)
    }
  }

  async function handleTestWebhook(webhookId: string) {
    setBusyWebhookId(webhookId)
    setActionError(null)
    setMessage(null)
    try {
      const result = await testWebhookSubscription(webhookId)
      setMessage(result.success ? 'Test webhook delivered' : 'Test webhook failed')
      await refreshAll()
    } catch (err) {
      setMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to test webhook')
    } finally {
      setBusyWebhookId(null)
    }
  }

  async function handleDeleteWebhook(webhookId: string) {
    setBusyWebhookId(webhookId)
    setActionError(null)
    setMessage(null)
    try {
      await deleteWebhookSubscription(webhookId)
      setSelectedWebhookId((current) => (current === webhookId ? null : current))
      setMessage('Webhook deleted')
      await refreshAll()
    } catch (err) {
      setMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to delete webhook')
    } finally {
      setBusyWebhookId(null)
    }
  }

  if (error) return <PageError error={error} onRetry={() => void refreshAll()} />

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Incident Alerts API</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-400">
            Configure outbound webhooks for churn incidents, signal changes, and durable artifact updates without leaving the product.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500" htmlFor="alerts-summary-window">Window</label>
          <select
            id="alerts-summary-window"
            value={summaryWindow}
            onChange={(event) => setSummaryWindow(Number(event.target.value) as (typeof SUMMARY_WINDOWS)[number])}
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
          >
            {SUMMARY_WINDOWS.map((days) => (
              <option key={days} value={days}>{days} days</option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void refreshAll()}
            disabled={refreshing}
            className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-800/60 hover:text-white disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {message ? (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
          {message}
        </div>
      ) : null}
      {actionError ? (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {actionError}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Active Webhooks"
          value={summary?.active_subscriptions ?? 0}
          icon={<BellRing className="h-4 w-4 text-cyan-400" />}
          skeleton={loading}
        />
        <StatCard
          label="Deliveries"
          value={summary?.total_deliveries ?? 0}
          icon={<FlaskConical className="h-4 w-4 text-violet-400" />}
          sub={`${summaryWindow}-day window`}
          skeleton={loading}
        />
        <StatCard
          label="Success Rate"
          value={formatPercent(summary?.success_rate)}
          icon={<CheckCircle2 className="h-4 w-4 text-emerald-400" />}
          sub={`${summary?.successful ?? 0} successful / ${summary?.failed ?? 0} failed`}
          skeleton={loading}
        />
        <StatCard
          label="Last Delivery"
          value={formatTs(summary?.last_delivery_at)}
          icon={<ShieldAlert className="h-4 w-4 text-amber-400" />}
          sub={`P95 ${formatDurationMs(summary?.p95_success_duration_ms)}`}
          skeleton={loading}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-medium text-white">Webhook Endpoints</h2>
              <p className="mt-1 text-sm text-slate-400">
                Use test delivery, toggle dispatch, and track 7-day health for each endpoint.
              </p>
            </div>
            <span className="rounded-full border border-slate-700 bg-slate-800/80 px-2 py-1 text-xs text-slate-300">
              {webhooks.length} configured
            </span>
          </div>

          <div className="mt-4 space-y-3">
            {webhooks.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/40 px-4 py-6 text-sm text-slate-400">
                No webhooks configured yet. Add a destination to turn churn incidents into external alerts.
              </div>
            ) : webhooks.map((webhook) => {
              const isBusy = busyWebhookId === webhook.id
              return (
                <article key={webhook.id} className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                    <div className="min-w-0 space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`rounded-full border px-2 py-1 text-xs ${channelTone(webhook.channel)}`}>
                          {CHANNEL_OPTIONS.find((option) => option.value === webhook.channel)?.label ?? webhook.channel}
                        </span>
                        <span className={`rounded-full border px-2 py-1 text-xs ${webhook.enabled
                          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
                          : 'border-slate-700 bg-slate-800/80 text-slate-300'}`}
                        >
                          {webhook.enabled ? 'enabled' : 'paused'}
                        </span>
                      </div>
                      <div className="truncate text-sm font-medium text-white">{webhook.url}</div>
                      {webhook.description ? (
                        <div className="text-sm text-slate-400">{webhook.description}</div>
                      ) : null}
                      <div className="flex flex-wrap gap-2">
                        {webhook.event_types.map((eventType) => (
                          <span key={eventType} className="rounded-full border border-slate-700 bg-slate-800/70 px-2 py-1 text-[11px] text-slate-300">
                            {eventType}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-xs text-slate-400 lg:min-w-[220px]">
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">7d deliveries</div>
                        <div className="mt-1 text-sm text-white">{webhook.recent_deliveries_7d}</div>
                      </div>
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">7d success</div>
                        <div className="mt-1 text-sm text-white">{formatPercent(webhook.recent_success_rate_7d)}</div>
                      </div>
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Created</div>
                        <div className="mt-1 text-sm text-white">{formatTs(webhook.created_at)}</div>
                      </div>
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Updated</div>
                        <div className="mt-1 text-sm text-white">{formatTs(webhook.updated_at)}</div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleTestWebhook(webhook.id)}
                      className="inline-flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-sm text-violet-200 transition-colors hover:bg-violet-500/20 disabled:opacity-50"
                    >
                      <FlaskConical className="h-4 w-4" />
                      Send Test
                    </button>
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleToggleWebhook(webhook.id, webhook.enabled)}
                      className="rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2 text-sm text-slate-200 transition-colors hover:bg-slate-700 disabled:opacity-50"
                    >
                      {webhook.enabled ? 'Pause Delivery' : 'Resume Delivery'}
                    </button>
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleDeleteWebhook(webhook.id)}
                      className="inline-flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200 transition-colors hover:bg-rose-500/20 disabled:opacity-50"
                    >
                      <Trash2 className="h-4 w-4" />
                      Delete
                    </button>
                    <button
                      type="button"
                      onClick={() => setSelectedWebhookId((current) => (current === webhook.id ? null : webhook.id))}
                      className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 transition-colors hover:bg-slate-800"
                    >
                      {selectedWebhookId === webhook.id ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                      {selectedWebhookId === webhook.id ? 'Hide Activity' : 'View Activity'}
                    </button>
                  </div>

                  {selectedWebhookId === webhook.id ? (
                    <div className="mt-4 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <h3 className="text-sm font-medium text-white">Recent Activity</h3>
                          <p className="mt-1 text-xs text-slate-400">
                            Delivery attempts and downstream CRM pushes for this endpoint.
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            refreshDeliveries()
                            refreshCrmPushLog()
                          }}
                          disabled={activityRefreshing}
                          className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-300 transition-colors hover:bg-slate-800/60 hover:text-white disabled:opacity-50"
                        >
                          <RefreshCw className={`h-3.5 w-3.5 ${activityRefreshing ? 'animate-spin' : ''}`} />
                          Refresh Activity
                        </button>
                      </div>

                      <div className="mt-4 grid gap-4 xl:grid-cols-2">
                        <div>
                          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Deliveries</div>
                          {activityLoading ? (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-4 text-sm text-slate-400">
                              Loading delivery activity...
                            </div>
                          ) : deliveryData?.deliveries?.length ? (
                            <div className="space-y-2">
                              {deliveryData.deliveries.map((delivery) => (
                                <div key={delivery.id} className={`rounded-lg border px-3 py-3 text-sm ${deliveryTone(delivery.success)}`}>
                                  <div className="flex items-center justify-between gap-3">
                                    <span className="font-medium">{delivery.event_type}</span>
                                    <span className="text-xs">
                                      {delivery.success ? 'success' : `failed${delivery.status_code ? ` · ${delivery.status_code}` : ''}`}
                                    </span>
                                  </div>
                                  <div className="mt-1 text-xs text-slate-300">
                                    attempt {delivery.attempt} · {formatDurationMs(delivery.duration_ms)} · {formatTs(delivery.delivered_at)}
                                  </div>
                                  {delivery.error ? (
                                    <div className="mt-2 text-xs text-rose-200">{delivery.error}</div>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-4 text-sm text-slate-400">
                              No recent delivery attempts for this webhook.
                            </div>
                          )}
                        </div>

                        <div>
                          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">CRM Push Log</div>
                          {selectedWebhook?.channel.startsWith('crm_') ? (
                            crmPushData?.pushes?.length ? (
                              <div className="space-y-2">
                                {crmPushData.pushes.map((push) => (
                                  <div key={push.id} className={`rounded-lg border px-3 py-3 text-sm ${deliveryTone(push.status === 'success')}`}>
                                    <div className="flex items-center justify-between gap-3">
                                      <span className="font-medium">
                                        {push.company_name || push.vendor_name || push.signal_type}
                                      </span>
                                      <span className="text-xs">{push.status}</span>
                                    </div>
                                    <div className="mt-1 text-xs text-slate-300">
                                      {push.crm_record_type || 'record'}{push.crm_record_id ? ` · ${push.crm_record_id}` : ''} · {formatTs(push.pushed_at)}
                                    </div>
                                    {push.error ? (
                                      <div className="mt-2 text-xs text-rose-200">{push.error}</div>
                                    ) : null}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-4 text-sm text-slate-400">
                                No CRM push activity recorded for this webhook yet.
                              </div>
                            )
                          ) : (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-4 text-sm text-slate-400">
                              CRM push history is only available for CRM webhook channels.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </article>
              )
            })}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <h2 className="text-sm font-medium text-white">Add Webhook</h2>
          <p className="mt-1 text-sm text-slate-400">
            Start with a generic JSON endpoint, or switch to Slack, Teams, or CRM-specific delivery.
          </p>

          <form className="mt-4 space-y-4" onSubmit={handleCreateWebhook}>
            <div>
              <label htmlFor="webhook-url" className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                Endpoint URL
              </label>
              <input
                id="webhook-url"
                value={form.url}
                onChange={(event) => setFormValue('url', event.target.value)}
                placeholder="https://hooks.example.com/atlas"
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="webhook-channel" className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Channel
                </label>
                <select
                  id="webhook-channel"
                  value={form.channel}
                  onChange={(event) => setFormValue('channel', event.target.value as WebhookChannel)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                >
                  {CHANNEL_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="webhook-description" className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Description
                </label>
                <input
                  id="webhook-description"
                  value={form.description}
                  onChange={(event) => setFormValue('description', event.target.value)}
                  placeholder="PagerDuty bridge"
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500"
                />
              </div>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between gap-3">
                <label htmlFor="webhook-secret" className="block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Signing Secret
                </label>
                <button
                  type="button"
                  onClick={() => setFormValue('secret', generateWebhookSecret())}
                  className="text-xs text-cyan-300 hover:text-cyan-200"
                >
                  Regenerate
                </button>
              </div>
              <input
                id="webhook-secret"
                value={form.secret}
                onChange={(event) => setFormValue('secret', event.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
              />
            </div>

            {requiresAuthHeader ? (
              <div>
                <label htmlFor="webhook-auth-header" className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
                  Auth Header
                </label>
                <input
                  id="webhook-auth-header"
                  value={form.auth_header}
                  onChange={(event) => setFormValue('auth_header', event.target.value)}
                  placeholder="Bearer pat-..."
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500"
                />
              </div>
            ) : null}

            <fieldset>
              <legend className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                Event Types
              </legend>
              <div className="grid gap-2">
                {EVENT_TYPE_OPTIONS.map((option) => (
                  <label
                    key={option.value}
                    className="flex items-start gap-3 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-3 text-sm text-slate-200"
                  >
                    <input
                      type="checkbox"
                      checked={sortedEvents.has(option.value)}
                      onChange={() => toggleEventType(option.value)}
                      className="mt-1 rounded border-slate-600 bg-slate-900 text-cyan-400"
                    />
                    <span>
                      <span className="block font-medium text-white">{option.label}</span>
                      <span className="mt-1 block text-xs text-slate-400">{option.description}</span>
                    </span>
                  </label>
                ))}
              </div>
              <div className="mt-2 text-xs text-slate-500">
                {selectedEventCount} event type{selectedEventCount === 1 ? '' : 's'} selected
              </div>
            </fieldset>

            <button
              type="submit"
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 transition-colors hover:bg-cyan-400 disabled:opacity-50"
            >
              <BellRing className="h-4 w-4" />
              {saving ? 'Adding Webhook...' : 'Add Webhook'}
            </button>
          </form>
        </section>
      </div>
    </div>
  )
}
