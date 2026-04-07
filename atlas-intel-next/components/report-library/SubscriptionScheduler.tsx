"use client";

import Link from 'next/link'
import { useMemo, useState } from 'react'
import {
  AlertCircle,
  Bell,
  CalendarClock,
  CheckCircle2,
  CreditCard,
  ExternalLink,
  Loader2,
  Mail,
  Save,
  Settings2,
} from 'lucide-react'
import {
  createBillingCheckout,
  createBillingPortal,
  fetchReportSubscription,
  upsertReportSubscription,
} from '@/lib/api/client'
import { useAuth, type User } from '@/lib/auth/AuthContext'
import useApiData from '@/lib/hooks/useApiData'
import type {
  ReportSubscription,
  ReportSubscriptionDeliverableFocus,
  ReportSubscriptionFrequency,
  ReportSubscriptionFreshnessPolicy,
  ReportSubscriptionScopeType,
} from '@/lib/types'

interface SubscriptionSchedulerProps {
  scopeType: ReportSubscriptionScopeType
  scopeKey: string
  scopeLabel: string
  description?: string
  defaultFocus?: ReportSubscriptionDeliverableFocus
  className?: string
}

interface SubscriptionSchedulerEditorProps extends SubscriptionSchedulerProps {
  initialLoad: boolean
  initialSubscription: ReportSubscription | null
  loadError: Error | null
  user: User | null
}

interface SubscriptionFormState {
  deliveryFrequency: ReportSubscriptionFrequency
  deliverableFocus: ReportSubscriptionDeliverableFocus
  freshnessPolicy: ReportSubscriptionFreshnessPolicy
  recipients: string
  deliveryNote: string
  enabled: boolean
}

const PLAN_LABELS: Record<string, string> = {
  b2b_trial: 'Trial',
  b2b_starter: 'Starter',
  b2b_growth: 'Growth',
  b2b_pro: 'Pro',
  trial: 'Trial',
  starter: 'Starter',
  growth: 'Growth',
  pro: 'Pro',
}

const RECIPIENT_EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function emptyFormState(
  defaultFocus: ReportSubscriptionDeliverableFocus,
): SubscriptionFormState {
  return {
    deliveryFrequency: 'weekly',
    deliverableFocus: defaultFocus,
    freshnessPolicy: 'fresh_or_monitor',
    recipients: '',
    deliveryNote: '',
    enabled: true,
  }
}

function formStateFromSubscription(
  subscription: ReportSubscription,
): SubscriptionFormState {
  return {
    deliveryFrequency: subscription.delivery_frequency,
    deliverableFocus: subscription.deliverable_focus,
    freshnessPolicy: subscription.freshness_policy,
    recipients: subscription.recipient_emails.join(', '),
    deliveryNote: subscription.delivery_note,
    enabled: subscription.enabled,
  }
}

function parseRecipients(raw: string): string[] {
  return raw
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean)
}

function frequencyLabel(value: ReportSubscriptionFrequency): string {
  if (value === 'weekly') return 'Weekly'
  if (value === 'monthly') return 'Monthly'
  return 'Quarterly'
}

function focusLabel(value: ReportSubscriptionDeliverableFocus): string {
  if (value === 'battle_cards') return 'Battle cards'
  if (value === 'executive_reports') return 'Executive reports'
  if (value === 'comparison_packs') return 'Comparison packs'
  return 'All deliverables'
}

function freshnessLabel(value: ReportSubscriptionFreshnessPolicy): string {
  if (value === 'fresh_only') return 'Fresh only'
  if (value === 'fresh_or_monitor') return 'Fresh + monitor'
  return 'Any freshness state'
}

function deliveryStatusLabel(status: ReportSubscription['last_delivery_status']): string {
  if (status === 'sent') return 'Delivered'
  if (status === 'partial') return 'Partial delivery'
  if (status === 'skipped') return 'Skipped'
  if (status === 'failed') return 'Failed'
  return 'No deliveries yet'
}

function deliveryStatusTone(status: ReportSubscription['last_delivery_status']): string {
  if (status === 'sent') return 'bg-emerald-500/15 text-emerald-300'
  if (status === 'partial') return 'bg-amber-500/15 text-amber-300'
  if (status === 'skipped') return 'bg-slate-500/15 text-slate-300'
  if (status === 'failed') return 'bg-rose-500/15 text-rose-300'
  return 'bg-slate-500/15 text-slate-300'
}

function planLabel(plan: string | undefined): string {
  if (!plan) return 'Account'
  return PLAN_LABELS[plan] ?? plan
}

function hasBilling(user: User | null): boolean {
  if (!user) return false
  return user.plan_status === 'active' || user.plan_status === 'past_due'
}

function canManageBilling(user: User | null): boolean {
  if (!user) return false
  return user.role === 'owner' || user.role === 'admin'
}

function recommendedUpgradePlan(user: User | null): string | null {
  if (!user) return null
  if (user.plan === 'b2b_growth' || user.plan === 'b2b_pro' || user.plan === 'growth' || user.plan === 'pro') {
    return null
  }

  const isB2BPlan = user.plan.startsWith('b2b_') || user.product.startsWith('b2b')
  return isB2BPlan ? 'b2b_growth' : 'growth'
}

function SubscriptionSchedulerEditor({
  scopeType,
  scopeKey,
  scopeLabel,
  description,
  defaultFocus = 'all',
  className = '',
  initialLoad,
  initialSubscription,
  loadError,
  user,
}: SubscriptionSchedulerEditorProps) {
  const [formState, setFormState] = useState<SubscriptionFormState>(() => (
    initialSubscription
      ? formStateFromSubscription(initialSubscription)
      : emptyFormState(defaultFocus)
  ))
  const [savedSubscription, setSavedSubscription] = useState<ReportSubscription | null>(initialSubscription)
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle')
  const [saveError, setSaveError] = useState('')
  const [billingState, setBillingState] = useState<'idle' | 'checkout' | 'portal'>('idle')
  const [billingError, setBillingError] = useState('')

  const recipientList = useMemo(
    () => parseRecipients(formState.recipients),
    [formState.recipients],
  )
  const recipientCount = recipientList.length
  const summary = `${frequencyLabel(formState.deliveryFrequency)} ${focusLabel(formState.deliverableFocus).toLowerCase()} for ${recipientCount || 1} recipient${recipientCount === 1 ? '' : 's'}`

  const billingEnabled = hasBilling(user)
  const selfServeBilling = canManageBilling(user)
  const upgradePlan = recommendedUpgradePlan(user)
  const primaryBillingActionLabel = billingEnabled && selfServeBilling
    ? 'Manage billing'
    : upgradePlan && selfServeBilling
      ? `Upgrade to ${planLabel(upgradePlan)}`
      : 'Open account'

  async function handleSave() {
    const invalidRecipients = recipientList.filter((value) => !RECIPIENT_EMAIL_PATTERN.test(value))
    if (invalidRecipients.length > 0) {
      setSaveError(`Enter valid recipient emails. Invalid: ${invalidRecipients.join(', ')}`)
      setSaveState('idle')
      return
    }
    if (formState.enabled && recipientList.length === 0) {
      setSaveError('Add at least one recipient before enabling recurring delivery.')
      setSaveState('idle')
      return
    }

    setSaveError('')
    setSaveState('saving')
    try {
      const result = await upsertReportSubscription(scopeType, scopeKey, {
        scope_label: scopeLabel,
        delivery_frequency: formState.deliveryFrequency,
        deliverable_focus: formState.deliverableFocus,
        freshness_policy: formState.freshnessPolicy,
        recipients: recipientList,
        delivery_note: formState.deliveryNote.trim(),
        enabled: formState.enabled,
      })
      setSavedSubscription(result.subscription)
      setFormState(formStateFromSubscription(result.subscription))
      setSaveState('saved')
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Could not save recurring delivery settings')
      setSaveState('idle')
    }
  }

  async function handleBillingAction() {
    setBillingError('')

    if (!user || !selfServeBilling) {
      window.location.href = '/account'
      return
    }

    try {
      if (billingEnabled) {
        setBillingState('portal')
        const data = await createBillingPortal({
          return_url: window.location.href,
        })
        window.location.href = data.portal_url
        return
      }

      if (upgradePlan) {
        setBillingState('checkout')
        const returnUrl = window.location.href
        const data = await createBillingCheckout({
          plan: upgradePlan,
          success_url: returnUrl,
          cancel_url: returnUrl,
        })
        window.location.href = data.checkout_url
        return
      }

      window.location.href = '/account'
    } catch (err) {
      setBillingError(err instanceof Error ? err.message : 'Could not open billing management')
      setBillingState('idle')
    }
  }

  return (
    <section
      id={`subscription-${scopeKey}`}
      className={`rounded-3xl border border-slate-700/60 bg-slate-900/60 p-5 sm:p-6 ${className}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.24em] text-cyan-300">
            <Bell className="h-3.5 w-3.5" />
            Recurring Delivery
          </div>
          <h2 className="mt-4 text-xl font-semibold text-white">{scopeLabel}</h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">
            {description ?? 'Persist delivery cadence, recipients, and trust constraints for this artifact scope.'}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-700/50 bg-slate-950/40 p-3 text-right">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Current Plan</p>
          <p className="mt-1 text-sm font-medium text-white">{user ? planLabel(user.plan) : 'Account'}</p>
          <p className="mt-1 text-xs text-slate-500">
            {formState.enabled ? 'Recurring delivery enabled' : 'Recurring delivery paused'}
          </p>
        </div>
      </div>

      {(loadError || saveError || billingError) && (
        <div className="mt-4 flex items-start gap-2 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{saveError || billingError || (loadError instanceof Error ? loadError.message : 'Could not load recurring delivery settings')}</span>
        </div>
      )}

      <div className={`mt-6 space-y-5 ${initialLoad ? 'opacity-70' : ''}`}>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              <CalendarClock className="h-3.5 w-3.5" />
              Delivery cadence
            </span>
            <select
              value={formState.deliveryFrequency}
              onChange={(event) => {
                setFormState((current) => ({ ...current, deliveryFrequency: event.target.value as ReportSubscriptionFrequency }))
                setSaveState('idle')
              }}
              disabled={initialLoad}
              className="w-full rounded-xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-cyan-500/50 disabled:opacity-60"
            >
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
              <option value="quarterly">Quarterly</option>
            </select>
          </label>

          <label className="block">
            <span className="mb-1.5 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              <Settings2 className="h-3.5 w-3.5" />
              Deliverable focus
            </span>
            <select
              value={formState.deliverableFocus}
              onChange={(event) => {
                setFormState((current) => ({ ...current, deliverableFocus: event.target.value as ReportSubscriptionDeliverableFocus }))
                setSaveState('idle')
              }}
              disabled={initialLoad}
              className="w-full rounded-xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-cyan-500/50 disabled:opacity-60"
            >
              <option value="all">All deliverables</option>
              <option value="battle_cards">Battle cards</option>
              <option value="executive_reports">Executive reports</option>
              <option value="comparison_packs">Comparison packs</option>
            </select>
          </label>

          <label className="block">
            <span className="mb-1.5 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Freshness gate</span>
            <select
              value={formState.freshnessPolicy}
              onChange={(event) => {
                setFormState((current) => ({ ...current, freshnessPolicy: event.target.value as ReportSubscriptionFreshnessPolicy }))
                setSaveState('idle')
              }}
              disabled={initialLoad}
              className="w-full rounded-xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-cyan-500/50 disabled:opacity-60"
            >
              <option value="fresh_only">Fresh only</option>
              <option value="fresh_or_monitor">Fresh + monitor</option>
              <option value="any">Any freshness state</option>
            </select>
          </label>

          <label className="block">
            <span className="mb-1.5 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              <Mail className="h-3.5 w-3.5" />
              Recipients
            </span>
            <input
              value={formState.recipients}
              onChange={(event) => {
                setFormState((current) => ({ ...current, recipients: event.target.value }))
                setSaveState('idle')
              }}
              disabled={initialLoad}
              placeholder="team@company.com, pmm@company.com"
              className="w-full rounded-xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50 disabled:opacity-60"
            />
          </label>
        </div>

        <label className="block">
          <span className="mb-1.5 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Delivery note</span>
          <textarea
            value={formState.deliveryNote}
            onChange={(event) => {
              setFormState((current) => ({ ...current, deliveryNote: event.target.value }))
              setSaveState('idle')
            }}
            disabled={initialLoad}
            rows={3}
            placeholder="Example: Include battle cards with sales-ready quality only, plus one executive summary slide."
            className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50 disabled:opacity-60"
          />
        </label>

        <label className="flex items-start gap-3 rounded-2xl border border-slate-700/60 bg-slate-950/30 px-4 py-3">
          <input
            type="checkbox"
            checked={formState.enabled}
            onChange={(event) => {
              setFormState((current) => ({ ...current, enabled: event.target.checked }))
              setSaveState('idle')
            }}
            disabled={initialLoad}
            className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-950 text-cyan-400 focus:ring-cyan-500/40"
          />
          <span>
            <span className="block text-sm font-medium text-white">Enable recurring delivery</span>
            <span className="mt-1 block text-sm leading-6 text-slate-400">
              Keep this schedule active for stakeholder drops and exported brief packages. Pause it without deleting saved recipients or trust rules.
            </span>
          </span>
        </label>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-3">
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Current Policy</p>
          <p className="mt-1 text-sm text-white">{summary}</p>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-3">
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Freshness</p>
          <p className="mt-1 text-sm text-white">{freshnessLabel(formState.freshnessPolicy)}</p>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-3">
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Delivery Status</p>
          <p className="mt-1 text-sm text-white">
            {savedSubscription ? deliveryStatusLabel(savedSubscription.last_delivery_status) : 'No saved rules yet'}
          </p>
          {savedSubscription?.last_delivery_report_count ? (
            <p className="mt-1 text-xs text-slate-500">
              Last package included {savedSubscription.last_delivery_report_count} artifact{savedSubscription.last_delivery_report_count === 1 ? '' : 's'}.
            </p>
          ) : null}
        </div>
      </div>

      <div className="mt-5 rounded-2xl border border-slate-700/60 bg-slate-950/30 p-4">
        <p className="text-sm font-medium text-white">Persistence</p>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          Delivery rules are stored durably per account and scope. This keeps the library view and report detail pages aligned on the same recurring-delivery policy instead of relying on browser-local state.
        </p>
      </div>

      {savedSubscription?.last_delivery_status && (
        <div className="mt-5 rounded-2xl border border-slate-700/60 bg-slate-950/30 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${deliveryStatusTone(savedSubscription.last_delivery_status)}`}>
              {deliveryStatusLabel(savedSubscription.last_delivery_status)}
            </span>
            {savedSubscription.last_delivery_at ? (
              <span className="text-xs text-slate-500">
                {new Date(savedSubscription.last_delivery_at).toLocaleString()}
              </span>
            ) : null}
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-300">
            {savedSubscription.last_delivery_summary || 'A recurring delivery attempt was recorded for this scope.'}
          </p>
          {savedSubscription.last_delivery_error ? (
            <p className="mt-2 text-sm leading-6 text-rose-300">
              {savedSubscription.last_delivery_error}
            </p>
          ) : null}
        </div>
      )}

      <div className="mt-5 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="text-xs leading-6 text-slate-500">
          {savedSubscription?.updated_at
            ? `Last saved ${new Date(savedSubscription.updated_at).toLocaleString()}`
            : initialLoad
              ? 'Loading saved delivery rules...'
              : 'No saved delivery rules for this scope yet.'}
          {savedSubscription?.next_delivery_at ? ` Next delivery ${new Date(savedSubscription.next_delivery_at).toLocaleString()}.` : ''}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {saveState === 'saved' && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
              <CheckCircle2 className="h-3.5 w-3.5" />
              Saved
            </span>
          )}
          <button
            onClick={handleSave}
            disabled={saveState === 'saving' || initialLoad}
            className="inline-flex items-center gap-2 rounded-xl bg-cyan-500/15 px-4 py-2 text-sm font-medium text-cyan-300 transition-colors hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saveState === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            {saveState === 'saving' ? 'Saving...' : 'Save delivery rules'}
          </button>
          <Link
            href="/account"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700/60 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:border-slate-600 hover:text-white"
          >
            Open account
            <ExternalLink className="h-4 w-4" />
          </Link>
          <button
            onClick={handleBillingAction}
            disabled={billingState !== 'idle'}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700/60 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            <CreditCard className="h-4 w-4" />
            {billingState === 'portal'
              ? 'Opening billing...'
              : billingState === 'checkout'
                ? 'Opening checkout...'
                : primaryBillingActionLabel}
          </button>
        </div>
      </div>
    </section>
  )
}

export default function SubscriptionScheduler(props: SubscriptionSchedulerProps) {
  const { user } = useAuth()
  const { data, loading, error } = useApiData(
    () => fetchReportSubscription(props.scopeType, props.scopeKey),
    [props.scopeType, props.scopeKey],
    {
      refreshOnFocus: false,
      refreshOnReconnect: false,
      minRefreshIntervalMs: 0,
    },
  )

  const initialSubscription = data?.subscription ?? null
  const editorKey = `${props.scopeType}:${props.scopeKey}:${initialSubscription?.updated_at ?? 'empty'}`

  return (
    <SubscriptionSchedulerEditor
      key={editorKey}
      {...props}
      user={user}
      initialLoad={loading && !data}
      initialSubscription={initialSubscription}
      loadError={error}
    />
  )
}
