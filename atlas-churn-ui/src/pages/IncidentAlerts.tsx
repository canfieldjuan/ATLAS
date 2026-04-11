import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { BellRing, CheckCircle2, ChevronDown, ChevronRight, Copy, FlaskConical, RefreshCw, ShieldAlert, Trash2 } from 'lucide-react'
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

const WEBHOOK_PRESETS: Array<{
  id: string
  label: string
  description: string
  channel: WebhookChannel
  event_types: WebhookEventType[]
}> = [
  {
    id: 'incident-response',
    label: 'Incident Response',
    description: 'High-signal churn and change alerts for paging or routing systems.',
    channel: 'generic',
    event_types: ['churn_alert', 'change_event'],
  },
  {
    id: 'evidence-monitoring',
    label: 'Evidence Monitoring',
    description: 'Ongoing signal and evidence refreshes for watchlist consumers.',
    channel: 'generic',
    event_types: ['change_event', 'signal_update'],
  },
  {
    id: 'artifact-delivery',
    label: 'Artifact Delivery',
    description: 'Notify downstream systems when new reports or battle cards are materialized.',
    channel: 'generic',
    event_types: ['report_generated'],
  },
  {
    id: 'crm-escalation',
    label: 'CRM Escalation',
    description: 'Push urgent churn incidents into CRM workflows with authenticated delivery.',
    channel: 'crm_hubspot',
    event_types: ['churn_alert', 'report_generated'],
  },
]

const CHANNEL_GUIDANCE: Record<WebhookChannel, { title: string; detail: string }> = {
  generic: {
    title: 'Signed JSON delivery',
    detail: 'Use any HTTPS endpoint that can verify the Atlas signing secret and accept the selected event payloads.',
  },
  slack: {
    title: 'Slack incoming webhook',
    detail: 'Point this at a Slack incoming webhook URL. Atlas formats the payload for Slack automatically.',
  },
  teams: {
    title: 'Teams workflow webhook',
    detail: 'Use a Microsoft Teams workflow or connector URL that accepts message-card style webhook payloads.',
  },
  crm_hubspot: {
    title: 'HubSpot CRM push',
    detail: 'Requires an auth header. Atlas will log downstream CRM pushes separately so delivery and CRM outcomes can be reviewed independently.',
  },
  crm_salesforce: {
    title: 'Salesforce CRM push',
    detail: 'Requires an auth header. Use this when churn incidents should create or update Salesforce records directly.',
  },
  crm_pipedrive: {
    title: 'Pipedrive CRM push',
    detail: 'Requires an auth header. Atlas will send authenticated CRM payloads and retain push history on this page.',
  },
}

const SUMMARY_WINDOWS = [7, 30, 90] as const
const MIN_SECRET_LENGTH = 16

const SAMPLE_EVENT_PAYLOADS: Record<WebhookEventType, Record<string, string | number>> = {
  churn_alert: {
    signal_type: 'competitive_displacement',
    severity: 'high',
    account_name: 'Acme Bank',
    company_name: 'Acme Bank',
    evidence_count: 4,
  },
  change_event: {
    event_type: 'pricing_change',
    delta: 2.1,
    market: 'mid-market support',
    confidence: 'elevated',
  },
  signal_update: {
    signal_type: 'review_velocity',
    evidence_window_days: 30,
    witness_count: 8,
    refresh_reason: 'watchlist_monitoring',
  },
  report_generated: {
    artifact_type: 'battle_card',
    report_title: 'Acme Rival Battle Card',
    report_id: 'report_123',
    freshness_state: 'fresh',
  },
}

function previewTimestamp() {
  return '2026-04-10T12:00:00Z'
}

function buildPreviewEnvelope(eventType: WebhookEventType) {
  return {
    event: eventType,
    vendor: 'Acme Rival',
    timestamp: previewTimestamp(),
    data: SAMPLE_EVENT_PAYLOADS[eventType],
  }
}

function formatSlackPreview(envelope: ReturnType<typeof buildPreviewEnvelope>) {
  const fields = Object.entries(envelope.data)
    .slice(0, 10)
    .map(([key, value]) => ({ type: 'mrkdwn', text: `*${key}:* ${value}` }))

  return {
    blocks: [
      { type: 'section', text: { type: 'mrkdwn', text: `:bell: *Atlas Intelligence Alert*  |  \`${envelope.event}\`` } },
      { type: 'section', text: { type: 'mrkdwn', text: `*Vendor:* ${envelope.vendor}  |  ${envelope.timestamp.slice(0, 19)}` } },
      ...(fields.length ? [{ type: 'section', fields }] : []),
    ],
    text: `Atlas: ${envelope.event} for ${envelope.vendor}`,
  }
}

function formatTeamsPreview(envelope: ReturnType<typeof buildPreviewEnvelope>) {
  const facts = Object.entries(envelope.data)
    .slice(0, 10)
    .map(([key, value]) => ({ title: String(key), value: String(value) }))

  return {
    type: 'message',
    attachments: [{
      contentType: 'application/vnd.microsoft.card.adaptive',
      contentUrl: null,
      content: {
        $schema: 'http://adaptivecards.io/schemas/adaptive-card.json',
        type: 'AdaptiveCard',
        version: '1.4',
        body: [
          {
            type: 'TextBlock',
            size: 'Medium',
            weight: 'Bolder',
            text: `Atlas Intelligence: ${envelope.event}`,
          },
          {
            type: 'FactSet',
            facts: [
              { title: 'Vendor', value: envelope.vendor },
              { title: 'Time', value: envelope.timestamp.slice(0, 19) },
              ...facts,
            ],
          },
        ],
      },
    }],
  }
}

function formatHubSpotPreview(envelope: ReturnType<typeof buildPreviewEnvelope>) {
  if (envelope.event === 'churn_alert' || envelope.event === 'signal_update') {
    const properties: Record<string, string> = {
      dealname: `Churn Signal: ${envelope.vendor}`,
      pipeline: 'default',
      dealstage: 'qualifiedtobuy',
      atlas_vendor: envelope.vendor,
      atlas_event_type: envelope.event,
    }
    Object.entries(envelope.data)
      .slice(0, 15)
      .forEach(([key, value]) => {
        properties[`atlas_${key}`.toLowerCase().replaceAll(' ', '_').slice(0, 50)] = String(value).slice(0, 65535)
      })
    return { properties }
  }

  return {
    properties: {
      hs_note_body: [
        `Atlas Intelligence: ${envelope.event} for ${envelope.vendor}`,
        ...Object.entries(envelope.data).slice(0, 20).map(([key, value]) => `  ${key}: ${value}`),
      ].join('\n'),
      hs_timestamp: envelope.timestamp,
    },
  }
}

function formatSalesforcePreview(envelope: ReturnType<typeof buildPreviewEnvelope>) {
  const fields: Record<string, string> = {
    Atlas_Vendor__c: envelope.vendor,
    Atlas_Event_Type__c: envelope.event,
    Atlas_Timestamp__c: envelope.timestamp,
  }
  Object.entries(envelope.data)
    .slice(0, 15)
    .forEach(([key, value]) => {
      fields[`Atlas_${key}__c`.slice(0, 40)] = String(value).slice(0, 255)
    })
  return fields
}

function formatPipedrivePreview(envelope: ReturnType<typeof buildPreviewEnvelope>) {
  if (envelope.event === 'churn_alert' || envelope.event === 'signal_update') {
    return {
      title: `Churn Signal: ${envelope.vendor}`,
      status: 'open',
      atlas_vendor: envelope.vendor,
      atlas_event_type: envelope.event,
      ...Object.fromEntries(
        Object.entries(envelope.data).slice(0, 10).map(([key, value]) => [`atlas_${key}`, String(value)]),
      ),
    }
  }

  return {
    content: [
      `Atlas Intelligence: ${envelope.event} for ${envelope.vendor}`,
      ...Object.entries(envelope.data).slice(0, 20).map(([key, value]) => `  ${key}: ${value}`),
    ].join('\n'),
  }
}

function buildPayloadPreview(channel: WebhookChannel, eventType: WebhookEventType) {
  const envelope = buildPreviewEnvelope(eventType)
  if (channel === 'slack') return formatSlackPreview(envelope)
  if (channel === 'teams') return formatTeamsPreview(envelope)
  if (channel === 'crm_hubspot') return formatHubSpotPreview(envelope)
  if (channel === 'crm_salesforce') return formatSalesforcePreview(envelope)
  if (channel === 'crm_pipedrive') return formatPipedrivePreview(envelope)
  return envelope
}

function validateWebhookForm(form: WebhookCreateBody) {
  const issues: string[] = []
  const url = form.url.trim()
  const secret = form.secret.trim()
  const authHeader = form.auth_header?.trim() || ''
  if (!url) {
    issues.push('Webhook URL is required')
  } else if (!/^https?:\/\//i.test(url)) {
    issues.push('Webhook URL must start with http:// or https://')
  }
  if (!secret) {
    issues.push('Webhook secret is required')
  } else if (secret.length < MIN_SECRET_LENGTH) {
    issues.push(`Webhook secret must be at least ${MIN_SECRET_LENGTH} characters`)
  }
  if (form.event_types.length === 0) {
    issues.push('Select at least one event type')
  }
  if (form.channel.startsWith('crm_') && !authHeader) {
    issues.push('CRM channels require an auth header')
  }
  return issues
}

function buildPreviewCurl(url: string, headers: Record<string, string>, payload: unknown) {
  const endpoint = url.trim() || 'https://hooks.example.com/atlas'
  const headerFlags = Object.entries(headers)
    .map(([key, value]) => `  -H '${key}: ${value.replaceAll("'", "'\\''")}'`)
    .join(' \\\n')
  const body = JSON.stringify(payload, null, 2).replaceAll("'", "'\\''")
  return `curl -X POST '${endpoint}' \\\n${headerFlags} \\\n  --data-raw '${body}'`
}

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

function formatFailureSummary(webhook: {
  latest_failure_status_code?: number | null
  latest_failure_error?: string | null
}) {
  if (webhook.latest_failure_error?.trim()) return webhook.latest_failure_error.trim()
  if (webhook.latest_failure_status_code != null) return `HTTP ${webhook.latest_failure_status_code}`
  return 'Delivery failed'
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

function buildActivitySearchParams({
  webhookId,
  deliveryStatus,
  deliveryEvent,
  crmStatus,
}: {
  webhookId: string
  deliveryStatus: 'all' | 'success' | 'failed'
  deliveryEvent: 'all' | WebhookEventType
  crmStatus: 'all' | 'success' | 'failed'
}) {
  const next = new URLSearchParams()
  next.set('webhook', webhookId)
  if (deliveryStatus !== 'all') next.set('delivery_status', deliveryStatus)
  if (deliveryEvent !== 'all') next.set('delivery_event', deliveryEvent)
  if (crmStatus !== 'all') next.set('crm_status', crmStatus)
  return next
}

export default function IncidentAlerts() {
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedWebhookId = searchParams.get('webhook')?.trim() || ''
  const requestedDeliveryStatus = searchParams.get('delivery_status') === 'success' || searchParams.get('delivery_status') === 'failed'
    ? (searchParams.get('delivery_status') as 'success' | 'failed')
    : 'all'
  const requestedDeliveryEvent = searchParams.get('delivery_event')?.trim() || 'all'
  const requestedCrmStatus = searchParams.get('crm_status') === 'success' || searchParams.get('crm_status') === 'failed'
    ? (searchParams.get('crm_status') as 'success' | 'failed')
    : 'all'
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
  const [selectedWebhookId, setSelectedWebhookId] = useState<string | null>(requestedWebhookId || null)
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null)
  const [selectedPreviewEventType, setSelectedPreviewEventType] = useState<WebhookEventType>('churn_alert')
  const [deliveryStatusFilter, setDeliveryStatusFilter] = useState<'all' | 'success' | 'failed'>(requestedDeliveryStatus)
  const [deliveryEventFilter, setDeliveryEventFilter] = useState<'all' | WebhookEventType>(requestedDeliveryEvent as 'all' | WebhookEventType)
  const [crmStatusFilter, setCrmStatusFilter] = useState<'all' | 'success' | 'failed'>(requestedCrmStatus)
  const [manualTestResults, setManualTestResults] = useState<Record<string, { success: boolean; testedAt: string }>>({})
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
  const validationIssues = useMemo(() => validateWebhookForm(form), [form])
  const previewEventType = form.event_types.includes(selectedPreviewEventType)
    ? selectedPreviewEventType
    : (form.event_types[0] ?? 'churn_alert')
  const previewPayload = useMemo(
    () => buildPayloadPreview(form.channel, previewEventType),
    [form.channel, previewEventType],
  )
  const previewHeaders = useMemo(() => ({
    'Content-Type': 'application/json',
    'X-Atlas-Event': previewEventType,
    'X-Atlas-Signature': 'sha256=<computed-hmac>',
    ...(form.auth_header?.trim() ? { Authorization: form.auth_header.trim() } : {}),
  }), [form.auth_header, previewEventType])
  const previewBodyText = useMemo(() => JSON.stringify(previewPayload, null, 2), [previewPayload])
  const previewCurl = useMemo(
    () => buildPreviewCurl(form.url, previewHeaders, previewPayload),
    [form.url, previewHeaders, previewPayload],
  )
  const activityLoading = deliveryLoading || crmPushLoading
  const activityRefreshing = deliveryRefreshing || crmPushRefreshing
  const channelGuidance = CHANNEL_GUIDANCE[form.channel]
  const filteredDeliveries = useMemo(() => {
    const deliveries = deliveryData?.deliveries ?? []
    return deliveries.filter((delivery) => {
      if (deliveryStatusFilter === 'success' && !delivery.success) return false
      if (deliveryStatusFilter === 'failed' && delivery.success) return false
      if (deliveryEventFilter !== 'all' && delivery.event_type !== deliveryEventFilter) return false
      return true
    })
  }, [deliveryData?.deliveries, deliveryEventFilter, deliveryStatusFilter])
  const filteredCrmPushes = useMemo(() => {
    const pushes = crmPushData?.pushes ?? []
    return pushes.filter((push) => {
      if (crmStatusFilter === 'success') return push.status === 'success'
      if (crmStatusFilter === 'failed') return push.status !== 'success'
      return true
    })
  }, [crmPushData?.pushes, crmStatusFilter])
  const deliveryEventOptions = useMemo(() => {
    const values = new Set((deliveryData?.deliveries ?? []).map((delivery) => delivery.event_type as WebhookEventType))
    return Array.from(values)
  }, [deliveryData?.deliveries])

  useEffect(() => {
    if (!requestedWebhookId) return
    if (!webhooks.some((webhook) => webhook.id === requestedWebhookId)) return
    setSelectedWebhookId((current) => (current === requestedWebhookId ? current : requestedWebhookId))
  }, [requestedWebhookId, webhooks])

  useEffect(() => {
    if (!selectedWebhookId) {
      if (!searchParams.has('webhook') && !searchParams.has('delivery_status') && !searchParams.has('delivery_event') && !searchParams.has('crm_status')) {
        return
      }
      setSearchParams((current) => {
        const next = new URLSearchParams(current)
        next.delete('webhook')
        next.delete('delivery_status')
        next.delete('delivery_event')
        next.delete('crm_status')
        return next
      }, { replace: true })
      return
    }

    const next = buildActivitySearchParams({
      webhookId: selectedWebhookId,
      deliveryStatus: deliveryStatusFilter,
      deliveryEvent: deliveryEventFilter,
      crmStatus: crmStatusFilter,
    })
    if (next.toString() === searchParams.toString()) return
    setSearchParams(next, { replace: true })
  }, [
    crmStatusFilter,
    deliveryEventFilter,
    deliveryStatusFilter,
    searchParams,
    selectedWebhookId,
    setSearchParams,
  ])

  async function refreshAll() {
    refreshSummary()
    refreshWebhooks()
    if (selectedWebhookId) {
      refreshDeliveries()
      refreshCrmPushLog()
    }
  }

  async function copyPreview(text: string, label: string) {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable in this browser')
      await navigator.clipboard.writeText(text)
      setActionError(null)
      setMessage(`Copied ${label}`)
    } catch (err) {
      setMessage(null)
      setActionError(err instanceof Error ? err.message : `Failed to copy ${label}`)
    }
  }

  async function copyActivityLink(webhookId: string) {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable in this browser')
      const next = buildActivitySearchParams({
        webhookId,
        deliveryStatus: selectedWebhookId === webhookId ? deliveryStatusFilter : 'all',
        deliveryEvent: selectedWebhookId === webhookId ? deliveryEventFilter : 'all',
        crmStatus: selectedWebhookId === webhookId ? crmStatusFilter : 'all',
      })
      const path = `${window.location.origin}${window.location.pathname}?${next.toString()}`
      await navigator.clipboard.writeText(path)
      setActionError(null)
      setMessage('Copied activity link')
    } catch (err) {
      setMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy activity link')
    }
  }

  function setFormValue<K extends keyof WebhookCreateBody>(key: K, value: WebhookCreateBody[K]) {
    setSelectedPresetId(null)
    setForm((current) => ({ ...current, [key]: value }))
  }

  function toggleEventType(eventType: WebhookEventType) {
    setSelectedPresetId(null)
    setForm((current) => {
      const next = new Set(current.event_types)
      if (next.has(eventType)) {
        next.delete(eventType)
      } else {
        next.add(eventType)
      }
      return { ...current, event_types: Array.from(next) as WebhookEventType[] }
    })
    setSelectedPreviewEventType(eventType)
  }

  function applyPreset(presetId: string) {
    const preset = WEBHOOK_PRESETS.find((item) => item.id === presetId)
    if (!preset) return
    setSelectedPresetId(preset.id)
    setSelectedPreviewEventType(preset.event_types[0] ?? 'churn_alert')
    setForm((current) => ({
      ...current,
      channel: preset.channel,
      event_types: preset.event_types,
    }))
  }

  async function handleCreateWebhook(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (validationIssues.length > 0) {
      setActionError(validationIssues[0])
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
      setSelectedPresetId(null)
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
      setManualTestResults((current) => ({
        ...current,
        [webhookId]: { success: result.success, testedAt: new Date().toISOString() },
      }))
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
              const latestManualTest = manualTestResults[webhook.id]
              const hasLatestFailure = Boolean(webhook.latest_failure_at)
              const testButtonLabel = hasLatestFailure ? 'Re-test Endpoint' : 'Send Test'
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
                      {webhook.latest_failure_at ? (
                        <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
                          <div className="font-medium text-rose-100">
                            Latest failure{webhook.latest_failure_event_type ? ` · ${webhook.latest_failure_event_type}` : ''}
                            {webhook.latest_failure_status_code != null ? ` · ${webhook.latest_failure_status_code}` : ''}
                          </div>
                          <div className="mt-1">
                            {formatFailureSummary(webhook)} · {formatTs(webhook.latest_failure_at)}
                          </div>
                        </div>
                      ) : null}
                      {latestManualTest ? (
                        <div className={`rounded-lg border px-3 py-2 text-xs ${
                          latestManualTest.success
                            ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-200'
                            : 'border-amber-500/20 bg-amber-500/10 text-amber-200'
                        }`}>
                          <div className="font-medium text-current">
                            Latest manual test {latestManualTest.success ? 'passed' : 'failed'}
                          </div>
                          <div className="mt-1">
                            {formatTs(latestManualTest.testedAt)}
                          </div>
                        </div>
                      ) : null}
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
                      {testButtonLabel}
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
                    <button
                      type="button"
                      onClick={() => void copyActivityLink(webhook.id)}
                      className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 transition-colors hover:bg-slate-800"
                    >
                      <Copy className="h-4 w-4" />
                      Copy Activity Link
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
                              <div className="mb-3 flex flex-wrap gap-2">
                                <select
                                  aria-label="Delivery status filter"
                                  value={deliveryStatusFilter}
                                  onChange={(event) => setDeliveryStatusFilter(event.target.value as 'all' | 'success' | 'failed')}
                                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                                >
                                  <option value="all">All delivery results</option>
                                  <option value="success">Successful only</option>
                                  <option value="failed">Failed only</option>
                                </select>
                                <select
                                  aria-label="Delivery event filter"
                                  value={deliveryEventFilter}
                                  onChange={(event) => setDeliveryEventFilter(event.target.value as 'all' | WebhookEventType)}
                                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                                >
                                  <option value="all">All event types</option>
                                  {deliveryEventOptions.map((eventType) => (
                                    <option key={eventType} value={eventType}>{eventType}</option>
                                  ))}
                                </select>
                              </div>
                              {filteredDeliveries.length ? filteredDeliveries.map((delivery) => (
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
                              )) : (
                                <div className="rounded-lg border border-dashed border-slate-700 bg-slate-950/50 px-3 py-4 text-sm text-slate-400">
                                  No deliveries match the current filters.
                                </div>
                              )}
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
                                <div className="mb-3 flex flex-wrap gap-2">
                                  <select
                                    aria-label="CRM push status filter"
                                    value={crmStatusFilter}
                                    onChange={(event) => setCrmStatusFilter(event.target.value as 'all' | 'success' | 'failed')}
                                    className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                                  >
                                    <option value="all">All CRM push results</option>
                                    <option value="success">Successful only</option>
                                    <option value="failed">Failed only</option>
                                  </select>
                                </div>
                                {filteredCrmPushes.length ? filteredCrmPushes.map((push) => (
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
                                )) : (
                                  <div className="rounded-lg border border-dashed border-slate-700 bg-slate-950/50 px-3 py-4 text-sm text-slate-400">
                                    No CRM pushes match the current filters.
                                  </div>
                                )}
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
            <fieldset>
              <legend className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                Presets
              </legend>
              <div className="grid gap-2">
                {WEBHOOK_PRESETS.map((preset) => {
                  const active = selectedPresetId === preset.id
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => applyPreset(preset.id)}
                      className={`rounded-lg border px-3 py-3 text-left transition-colors ${
                        active
                          ? 'border-cyan-500/40 bg-cyan-500/10'
                          : 'border-slate-800 bg-slate-950/50 hover:border-slate-700 hover:bg-slate-900/80'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm font-medium text-white">{preset.label}</span>
                        <span className="text-[11px] uppercase tracking-wide text-slate-400">
                          {CHANNEL_OPTIONS.find((option) => option.value === preset.channel)?.label ?? preset.channel}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-slate-400">{preset.description}</div>
                    </button>
                  )
                })}
              </div>
            </fieldset>

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
                <div className="mt-2 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
                  <div className="font-medium text-white">{channelGuidance.title}</div>
                  <div className="mt-1 text-slate-400">{channelGuidance.detail}</div>
                </div>
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

            <div className="grid gap-4 xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Setup Checks</div>
                <div className="mt-2 text-sm text-slate-300">
                  {validationIssues.length ? 'Resolve these issues before saving this webhook.' : 'This webhook configuration is ready to save.'}
                </div>
                <ul className="mt-3 space-y-2 text-sm">
                  {validationIssues.length ? validationIssues.map((issue) => (
                    <li key={issue} className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-rose-200">
                      {issue}
                    </li>
                  )) : (
                    <li className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-emerald-200">
                      Atlas will sign the payload, include the selected event header, and deliver the formatted body shown here.
                    </li>
                  )}
                </ul>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Payload Preview</div>
                    <div className="mt-1 text-sm text-slate-300">
                      Sample {previewEventType} payload for {CHANNEL_OPTIONS.find((option) => option.value === form.channel)?.label ?? form.channel}.
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <label htmlFor="preview-event-type" className="text-xs uppercase tracking-wide text-slate-500">
                      Preview Event
                    </label>
                    <select
                      id="preview-event-type"
                      value={previewEventType}
                      onChange={(event) => setSelectedPreviewEventType(event.target.value as WebhookEventType)}
                      className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                    >
                      {form.event_types.length ? form.event_types.map((eventType) => (
                        <option key={eventType} value={eventType}>{eventType}</option>
                      )) : (
                        <option value="churn_alert">churn_alert</option>
                      )}
                    </select>
                    <span className={`rounded-full border px-2 py-1 text-xs ${channelTone(form.channel)}`}>
                      {CHANNEL_OPTIONS.find((option) => option.value === form.channel)?.label ?? form.channel}
                    </span>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void copyPreview(previewBodyText, 'sample JSON')}
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 transition-colors hover:bg-slate-800"
                  >
                    <Copy className="h-3.5 w-3.5" />
                    Copy Sample JSON
                  </button>
                  <button
                    type="button"
                    onClick={() => void copyPreview(previewCurl, 'sample cURL')}
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 transition-colors hover:bg-slate-800"
                  >
                    <Copy className="h-3.5 w-3.5" />
                    Copy cURL
                  </button>
                </div>
                <div className="mt-4 grid gap-4 lg:grid-cols-2">
                  <div>
                    <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Headers Sent</div>
                    <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 px-3 py-3 text-xs text-slate-300">{JSON.stringify(previewHeaders, null, 2)}</pre>
                  </div>
                  <div>
                    <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Request Body</div>
                    <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 px-3 py-3 text-xs text-slate-300">{previewBodyText}</pre>
                  </div>
                </div>
              </div>
            </div>

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
