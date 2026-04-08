import { useState, useEffect, useCallback } from 'react'
import { X, Bell, Loader2, Check, AlertTriangle } from 'lucide-react'
import { clsx } from 'clsx'
import { fetchReportSubscription, upsertReportSubscription } from '../api/client'
import type { ReportSubscription, ReportSubscriptionUpsert } from '../api/client'

interface SubscriptionModalProps {
  open: boolean
  onClose: () => void
  scopeType: 'library' | 'report'
  scopeKey: string
  scopeLabel: string
  onSaved?: (sub: ReportSubscription) => void
}

export default function SubscriptionModal({
  open, onClose, scopeType, scopeKey, scopeLabel, onSaved,
}: SubscriptionModalProps) {
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

  // Load existing subscription
  useEffect(() => {
    if (!open) return
    setLoading(true)
    setError('')
    setSuccess(false)
    fetchReportSubscription(scopeType, scopeKey)
      .then(res => {
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
          setRecipients('')
        }
      })
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load subscription'))
      .finally(() => setLoading(false))
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

    setSaving(true)
    setError('')
    setSuccess(false)

    try {
      const body: ReportSubscriptionUpsert = {
        scope_label: label || scopeLabel,
        delivery_frequency: frequency,
        deliverable_focus: focus,
        freshness_policy: freshness,
        recipients: recipientList,
        delivery_note: note,
        enabled,
      }
      const res = await upsertReportSubscription(scopeType, scopeKey, body)
      setExisting(res.subscription)
      setSuccess(true)
      onSaved?.(res.subscription)
      setTimeout(() => onClose(), 1200)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save subscription')
    } finally {
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

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-slate-900 border border-slate-700/50 rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700/50">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-semibold text-white">
              {existing ? 'Manage Subscription' : 'Subscribe to Reports'}
            </h2>
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
            {/* Label */}
            <div>
              <label className="block text-xs text-slate-400 mb-1">Subscription Label</label>
              <input
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
                <label className="block text-xs text-slate-400 mb-1">Frequency</label>
                <select
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
                <label className="block text-xs text-slate-400 mb-1">Deliverable Focus</label>
                <select
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
              <label className="block text-xs text-slate-400 mb-1">Freshness Policy</label>
              <select
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
              <label className="block text-xs text-slate-400 mb-1">Recipient Emails</label>
              <textarea
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
              <label className="block text-xs text-slate-400 mb-1">Delivery Note (optional)</label>
              <input
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

            {/* Last delivery status */}
            {existing?.last_delivery_at && (
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30 text-xs text-slate-400">
                Last delivery: {new Date(existing.last_delivery_at).toLocaleDateString()}
                {existing.last_delivery_status && (
                  <span className={clsx(
                    'ml-2 px-1.5 py-0.5 rounded',
                    existing.last_delivery_status === 'sent' ? 'bg-green-900/40 text-green-300' :
                    existing.last_delivery_status === 'failed' ? 'bg-red-900/40 text-red-300' :
                    'bg-slate-700 text-slate-300',
                  )}>
                    {existing.last_delivery_status}
                  </span>
                )}
                {existing.last_delivery_report_count != null && (
                  <span className="ml-1">({existing.last_delivery_report_count} reports)</span>
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
              {saving ? 'Saving...' : existing ? 'Update' : 'Subscribe'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
