import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SubscriptionModal from './SubscriptionModal'

const api = vi.hoisted(() => ({
  fetchReportSubscription: vi.fn(),
  upsertReportSubscription: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('SubscriptionModal', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('resets form defaults when opening a scope without an existing subscription', async () => {
    const onClose = vi.fn()

    api.fetchReportSubscription
      .mockResolvedValueOnce({
        subscription: {
          id: 'sub-1',
          scope_type: 'report',
          scope_key: 'report-1',
          scope_label: 'Existing report sub',
          report_id: 'report-1',
          delivery_frequency: 'monthly',
          deliverable_focus: 'battle_cards',
          freshness_policy: 'any',
          recipient_emails: ['team@example.com'],
          delivery_note: 'Carry forward',
          enabled: false,
          next_delivery_at: null,
          last_delivery_at: null,
          last_delivery_status: null,
          last_delivery_report_count: null,
          last_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      })
      .mockResolvedValueOnce({ subscription: null })

    const { rerender } = render(
      <SubscriptionModal
        open
        onClose={onClose}
        scopeType="report"
        scopeKey="report-1"
        scopeLabel="Report One"
      />,
    )

    await waitFor(() => {
      expect(api.fetchReportSubscription).toHaveBeenCalledWith('report', 'report-1')
    })

    expect(screen.getByDisplayValue('Existing report sub')).toBeInTheDocument()
    expect(screen.getByLabelText('Frequency')).toHaveValue('monthly')
    expect(screen.getByLabelText('Deliverable Focus')).toHaveValue('battle_cards')
    expect(screen.getByLabelText('Freshness Policy')).toHaveValue('any')
    expect(screen.getByDisplayValue('team@example.com')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Carry forward')).toBeInTheDocument()
    expect(screen.getByText('Paused')).toBeInTheDocument()

    rerender(
      <SubscriptionModal
        open
        onClose={onClose}
        scopeType="library"
        scopeKey="library"
        scopeLabel="Full Report Library"
      />,
    )

    await waitFor(() => {
      expect(api.fetchReportSubscription).toHaveBeenCalledWith('library', 'library')
    })

    expect(screen.getByDisplayValue('Full Report Library')).toBeInTheDocument()
    expect(screen.getByLabelText('Frequency')).toHaveValue('weekly')
    expect(screen.getByLabelText('Deliverable Focus')).toHaveValue('all')
    expect(screen.getByLabelText('Freshness Policy')).toHaveValue('fresh_or_monitor')
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.queryByDisplayValue('team@example.com')).not.toBeInTheDocument()
    expect(screen.queryByDisplayValue('Carry forward')).not.toBeInTheDocument()
  })
})
