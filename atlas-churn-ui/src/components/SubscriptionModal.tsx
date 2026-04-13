import { useState, useEffect, useCallback, useRef } from 'react'
import { X, Bell, Loader2, Check, AlertTriangle, Calendar, Clock } from 'lucide-react'
import { clsx } from 'clsx'
import { fetchReportSubscription, upsertReportSubscription } from '../api/client'
import { normalizeReportLibraryViewFilters } from '../api/client'
import type {
  ReportLibraryViewFilters,
  ReportSubscription,
  ReportSubscriptionScopeType,
  ReportSubscriptionUpsert,
} from '../api/client'

interface SubscriptionModalProps {
  open: boolean
  onClose: () => void
  scopeType: ReportSubscriptionScopeType
  scopeKey: string
  scopeLabel: string
  filterPayload?: ReportLibraryViewFilters
  onSaved?: (sub: ReportSubscription) => void
}

function describeFilterPayload(filterPayload?: ReportLibraryViewFilters) {
  const normalized = normalizeReportLibraryViewFilters(filterPayload)
  const lines: string[] = []
  if (normalized.report_type) lines.push(`Type: ${normalized.report_type.replace(/_/g, ' ')}`)
  if (normalized.vendor_filter) lines.push(`Vendor: ${normalized.vendor_filter}`)
  if (normalized.quality_status) lines.push(`Quality: ${normalized.quality_status.replace(/_/g, ' ')}`)
  if (normalized.freshness_state) lines.push(`Freshness: ${normalized.freshness_state.replace(/_/g, ' ')}`)
  if (normalized.review_state) lines.push(`Review: ${normalized.review_state.replace(/_/g, ' ')}`)
  return lines
}

function describeSubscriptionScope(scopeType: ReportSubscriptionScopeType) {
  if (scopeType === 'report') return 'report'
  if (scopeType === 'library_view') return 'library view'
  return 'library'
}

function getSubscriptionModalPresentation(
  scopeType: ReportSubscriptionScopeType,
  existing: ReportSubscription | null,
  enabled: boolean,
) {
  const scopeLabel = describeSubscriptionScope(scopeType)
  if (!existing) {
    return {
      heading: `Subscribe to ${scopeLabel === 'report' ? 'Report' : 'Library'}`,
      badge: 'New',
      badgeClassName: 'bg-slate-800 text-slate-300 border border-slate-700',
      description: `Set up recurring delivery for this ${scopeLabel}.`,
    }
  }

  if (enabled) {
    return {
      heading: 'Manage Active Subscription',
      badge: 'Active',
      badgeClassName: 'bg-cyan-900/30 text-cyan-300 border border-cyan-700/40',
      description: 'This subscription is currently delivering on schedule.',
    }
  }

  return {
    heading: 'Resume Paused Subscription',
    badge: 'Paused',
    badgeClassName: 'bg-amber-500/10 text-amber-300 border border-amber-500/25',
    description: 'Delivery is paused until you reactivate this subscription.',
  }
}

function getSubscriptionSavePresentation(
  scopeType: ReportSubscriptionScopeType,
  existing: ReportSubscription | null,
  enabled: boolean,
) {
  const scopeLabel = describeSubscriptionScope(scopeType)
  if (!existing) {
    if (enabled) {
      return {
        actionLabel: 'Subscribe',
        savingLabel: 'Subscribing...',
        helperText: `This will create a recurring delivery for this ${scopeLabel}.`,
      }
    }
    return {
      actionLabel: 'Save Paused Subscription',
      savingLabel: 'Saving...',
      helperText: 'This will save the subscription without starting delivery yet.',
    }
  }

  if (existing.enabled && enabled) {
    return {
      actionLabel: 'Update Subscription',
      savingLabel: 'Saving...',
      helperText: 'This will update the active delivery schedule.',
    }
  }

  if (existing.enabled && !enabled) {
    return {
      actionLabel: 'Pause Subscription',
      savingLabel: 'Saving...',
      helperText: 'This will pause recurring delivery until you resume it.',
    }
  }

  if (!existing.enabled && enabled) {
    return {
      actionLabel: 'Resume Subscription',
      savingLabel: 'Saving...',
      helperText: 'This will reactivate recurring delivery with these settings.',
    }
  }

  return {
    actionLabel: 'Update Paused Subscription',
    savingLabel: 'Saving...',
    helperText: 'This will keep delivery paused while saving your changes.',
  }
}

export default function SubscriptionModal({
  open, onClose, scopeType, scopeKey, scopeLabel, filterPayload, onSaved,
}: SubscriptionModalProps) {
  const idPrefix = `subscription-${scopeType}-${scopeKey.replace(/[^a-zA-Z0-9_-]/g, '-')}`
  const loadRequestIdRef = useRef(0)
  const saveRequestIdRef = useRef(0)
  const saveCloseTimerRef = useRef<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  // Form state
  const [label, setLabel] = useState(scopeLabel)
  const [frequency, setFrequency] = useState<'weekly' | 'monthly' | 'quarterly'>('weekly')
  const [focus, setFocus] = useState<'all' | 'battle_cards' | 'executive_reports' | 'comparison_packs'>('all')
  const [freshness, setFreshness] = useState<'fresh_only' | 'fresh_or_monitor' | 'any'>('fresh_or_monitor')
  const [recipients, setRecipients] = useState('')
  const [note, setNote] = useState('')
  const [enabled, setEnabled] = useState(true)
  const [existing, setExisting] = useState<ReportSubscription | null>(null)
  const filterSummary = describeFilterPayload(filterPayload)
  const modalPresentation = getSubscriptionModalPresentation(scopeType, existing, enabled)
  const savePresentation = getSubscriptionSavePresentation(scopeType, existing, enabled)

  // Load existing subscription
  useEffect(() => {
    loadRequestIdRef.current += 1
    saveRequestIdRef.current += 1
    if (saveCloseTimerRef.current != null) {
      window.clearTimeout(saveCloseTimerRef.current)
      saveCloseTimerRef.current = null
    }
    setSaving(false)
    if (!open) {
      setLoading(false)
      return
    }
    const requestId = loadRequestIdRef.current
    setLoading(true)
    setError('')
    setSuccess(false)
    fetchReportSubscription(scopeType, scopeKey)
      .then(res => {
        if (loadRequestIdRef.current !== requestId) return
        const sub = res.subscription
        if (sub) {
          setExisting(sub)
          setLabel(sub.scope_label)
          setFrequency(sub.delivery_frequency)
          setFocus(sub.deliverable_focus)
          setFreshness(sub.freshness_policy)
          setRecipients(sub.recipient_emails.join(', '))
          setNote(sub.delivery_note || '')
          setEnabled(sub.enabled)
        } else {
          setExisting(null)
          setLabel(scopeLabel)
          setFrequency('weekly')
          setFocus('all')
          setFreshness('fresh_or_monitor')
          setRecipients('')
          setNote('')
          setEnabled(true)
        }
      })
      .catch(err => {
        if (loadRequestIdRef.current !== requestId) return
        setError(err instanceof Error ? err.message : 'Failed to load subscription')
      })
      .finally(() => {
        if (loadRequestIdRef.current !== requestId) return
        setLoading(false)
      })
  }, [open, scopeType, scopeKey, scopeLabel])

  const handleSave = async () => {
    const recipientList = recipients
      .split(/[,;\n]/)
      .map(e => e.trim().toLowerCase())
      .filter(e => e.includes('@'))

    if (enabled && recipientList.length === 0) {
      setError('Add at least one recipient email to enable delivery')
      return
    }

    const requestId = saveRequestIdRef.current + 1
    saveRequestIdRef.current = requestId
    if (saveCloseTimerRef.current != null) {
      window.clearTimeout(saveCloseTimerRef.current)
      saveCloseTimerRef.current = null
    }
    setSaving(true)
    setError('')
    setSuccess(false)

    try {
      const body: ReportSubscriptionUpsert = {
        scope_label: label || scopeLabel,
        filter_payload: scopeType === 'library_view' ? normalizeReportLibraryViewFilters(filterPayload) : undefined,
        delivery_frequency: frequency,
        deliverable_focus: focus,
        freshness_policy: freshness,
        recipients: recipientList,
        delivery_note: note,
        enabled,
      }
      const res = await upsertReportSubscription(scopeType, scopeKey, body)
      if (saveRequestIdRef.current !== requestId) return
      setExisting(res.subscription)
      setSuccess(true)
      saveCloseTimerRef.current = window.setTimeout(() => {
        if (saveRequestIdRef.current !== requestId) return
        onSaved?.(res.subscription)
        onClose()
      }, 1200)
    } catch (err) {
      if (saveRequestIdRef.current !== requestId) return
      setError(err instanceof Error ? err.message : 'Failed to save subscription')
    } finally {
      if (saveRequestIdRef.current !== requestId) return
      setSaving(false)
    }
  }

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    if (open) document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, handleKeyDown])

  useEffect(() => () => {
    loadRequestIdRef.current += 1
    saveRequestIdRef.current += 1
    if (saveCloseTimerRef.current != null) {
      window.clearTimeout(saveCloseTimerRef.current)
      saveCloseTimerRef.current = null
    }
  }, [])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-slate-900 border border-slate-700/50 rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700/50">
          <div className="flex items-start gap-3">
            <Bell className="w-5 h-5 text-cyan-400 mt-0.5" />
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-white">
                  {modalPresentation.heading}
                </h2>
                <span className={clsx('inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium', modalPresentation.badgeClassName)}>
                  {modalPresentation.badge}
                </span>
              </div>
              <p className="mt-1 text-sm text-slate-400">
                {modalPresentation.description}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
          </div>
        ) : (
          <div className="p-6 space-y-4">
            <div className={clsx('rounded-lg px-3 py-2 border', modalPresentation.badgeClassName)}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] opacity-80">Subscription Status</div>
              <div className="mt-1 text-sm">{modalPresentation.description}</div>
            </div>
            {scopeType === 'library_view' && filterSummary.length > 0 && (
              <div className="rounded-lg border border-cyan-800/40 bg-cyan-950/30 px-3 py-2">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan-300/80">Subscribed View</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {filterSummary.map((line) => (
                    <span
                      key={line}
                      className="rounded-full border border-cyan-700/40 bg-slate-900/70 px-2.5 py-1 text-xs text-cyan-100"
                    >
                      {line}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {/* Label */}
            <div>
              <label htmlFor={`${idPrefix}-label`} className="block text-xs text-slate-400 mb-1">Subscription Label</label>
              <input
                id={`${idPrefix}-label`}
                type="text"
                value={label}
                onChange={e => setLabel(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
                placeholder="e.g. Weekly Zendesk Intel"
              />
            </div>

            {/* Frequency + Focus */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor={`${idPrefix}-frequency`} className="block text-xs text-slate-400 mb-1">Frequency</label>
                <select
                  id={`${idPrefix}-frequency`}
                  value={frequency}
                  onChange={e => setFrequency(e.target.value as typeof frequency)}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
                >
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                  <option value="quarterly">Quarterly</option>
                </select>
              </div>
              <div>
                <label htmlFor={`${idPrefix}-focus`} className="block text-xs text-slate-400 mb-1">Deliverable Focus</label>
                <select
                  id={`${idPrefix}-focus`}
                  value={focus}
                  onChange={e => setFocus(e.target.value as typeof focus)}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
                >
                  <option value="all">All Reports</option>
                  <option value="battle_cards">Battle Cards</option>
                  <option value="executive_reports">Executive Reports</option>
                  <option value="comparison_packs">Comparison Packs</option>
                </select>
              </div>
            </div>

            {/* Freshness */}
            <div>
              <label htmlFor={`${idPrefix}-freshness`} className="block text-xs text-slate-400 mb-1">Freshness Policy</label>
              <select
                id={`${idPrefix}-freshness`}
                value={freshness}
                onChange={e => setFreshness(e.target.value as typeof freshness)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
              >
                <option value="fresh_only">Fresh only (skip if no new data)</option>
                <option value="fresh_or_monitor">Fresh or monitoring (include stale with label)</option>
                <option value="any">Any (always deliver)</option>
              </select>
            </div>

            {/* Recipients */}
            <div>
              <label htmlFor={`${idPrefix}-recipients`} className="block text-xs text-slate-400 mb-1">Recipient Emails</label>
              <textarea
                id={`${idPrefix}-recipients`}
                value={recipients}
                onChange={e => setRecipients(e.target.value)}
                rows={2}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none resize-none"
                placeholder="team@company.com, analyst@company.com"
              />
              <p className="text-xs text-slate-500 mt-0.5">Comma-separated email addresses</p>
            </div>

            {/* Note */}
            <div>
              <label htmlFor={`${idPrefix}-note`} className="block text-xs text-slate-400 mb-1">Delivery Note (optional)</label>
              <input
                id={`${idPrefix}-note`}
                type="text"
                value={note}
                onChange={e => setNote(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:border-cyan-500 focus:outline-none"
                placeholder="Internal context for this subscription"
              />
            </div>

            {/* Enabled toggle */}
            <label className="flex items-center gap-3 cursor-pointer">
              <div
                className={clsx(
                  'relative w-10 h-5 rounded-full transition-colors',
                  enabled ? 'bg-cyan-600' : 'bg-slate-700',
                )}
                onClick={() => setEnabled(!enabled)}
              >
                <div className={clsx(
                  'absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform',
                  enabled ? 'translate-x-5' : 'translate-x-0.5',
                )} />
              </div>
              <span className="text-sm text-slate-300">{enabled ? 'Active' : 'Paused'}</span>
            </label>

            {/* Delivery schedule & history */}
            {existing && (existing.next_delivery_at || existing.last_delivery_at) && (
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30 space-y-2">
                {existing.next_delivery_at && (
                  <div className="flex items-center gap-2 text-xs">
                    <Calendar className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="text-slate-400">Next delivery:</span>
                    <span className="text-white font-medium">
                      {new Date(existing.next_delivery_at).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}
                    </span>
                  </div>
                )}
                {existing.last_delivery_at && (
                  <div className="flex items-center gap-2 text-xs">
                    <Clock className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                    <span className="text-slate-400">Last delivery:</span>
                    <span className="text-slate-300">
                      {new Date(existing.last_delivery_at).toLocaleDateString()}
                    </span>
                    {existing.last_delivery_status && (
                      <span className={clsx(
                        'px-1.5 py-0.5 rounded',
                        existing.last_delivery_status === 'sent' ? 'bg-green-900/40 text-green-300' :
                        existing.last_delivery_status === 'partial' ? 'bg-amber-900/40 text-amber-300' :
                        existing.last_delivery_status === 'failed' ? 'bg-red-900/40 text-red-300' :
                        existing.last_delivery_status === 'skipped' ? 'bg-slate-700 text-slate-400' :
                        'bg-slate-700 text-slate-300',
                      )}>
                        {existing.last_delivery_status}
                      </span>
                    )}
                    {existing.last_delivery_report_count != null && (
                      <span className="text-slate-500">({existing.last_delivery_report_count} report{existing.last_delivery_report_count === 1 ? '' : 's'})</span>
                    )}
                  </div>
                )}
                {existing.last_delivery_summary && (
                  <p className="text-xs text-slate-500 pl-[1.375rem]">{existing.last_delivery_summary}</p>
                )}
              </div>
            )}

            {/* Error / Success */}
            {error && (
              <div className="flex items-center gap-2 text-sm text-red-400">
                <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
              </div>
            )}
            {success && (
              <div className="flex items-center gap-2 text-sm text-green-400">
                <Check className="w-4 h-4" /> Subscription saved
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        {!loading && (
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-slate-700/50">
            <p className="mr-auto text-xs text-slate-500">
              {savePresentation.helperText}
            </p>
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-600 text-white rounded-lg transition-colors"
            >
              {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Bell className="w-4 h-4" />}
              {saving ? savePresentation.savingLabel : savePresentation.actionLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
