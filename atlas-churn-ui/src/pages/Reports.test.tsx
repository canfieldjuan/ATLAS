import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { RouterProvider, createMemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Reports from './Reports'

const planGate = vi.hoisted(() => ({
  canAccessReports: true,
}))

const modalState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const api = vi.hoisted(() => ({
  buildReportLibraryViewScopeKey: vi.fn((filters?: Record<string, string>) => {
    return [
      'library-view',
      filters?.report_type || 'all',
      filters?.vendor_filter || 'all',
      filters?.quality_status || 'all',
    ].join('::')
  }),
  fetchReports: vi.fn(),
  listReportSubscriptions: vi.fn(),
  generateAccountComparisonReport: vi.fn(),
  generateAccountDeepDiveReport: vi.fn(),
  generateVendorComparisonReport: vi.fn(),
  requestBattleCardReport: vi.fn(),
}))

vi.mock('../api/client', () => api)
vi.mock('../hooks/usePlanGate', () => ({
  usePlanGate: () => planGate,
}))
vi.mock('../components/SubscriptionModal', () => ({
  default: (props: any) => {
    modalState.lastProps = props
    if (!props.open) return null
    return (
      <div data-testid="subscription-modal">
        {props.scopeType}:{props.scopeKey}:{props.scopeLabel}
      </div>
    )
  },
}))

describe('Reports', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    modalState.lastProps = null
    planGate.canAccessReports = true
    api.listReportSubscriptions.mockResolvedValue({ subscriptions: [] })
    api.fetchReports.mockResolvedValue({
      reports: [
        {
          id: 'report-1',
          report_type: 'battle_card',
          vendor_filter: 'Zendesk',
          category_filter: null,
          executive_summary: 'Summary',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'completed',
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: 'sales_ready',
          latest_failure_step: null,
          latest_error_summary: null,
        },
      ],
      count: 1,
    })
    api.generateVendorComparisonReport.mockResolvedValue({ execution_id: 'exec-1' })
    api.generateAccountComparisonReport.mockResolvedValue({ execution_id: 'exec-2' })
    api.generateAccountDeepDiveReport.mockResolvedValue({ execution_id: 'exec-3' })
    api.requestBattleCardReport.mockResolvedValue({ execution_id: 'exec-4' })
  })

  it('opens a filtered library subscription as a library_view scope', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Reports')

    fireEvent.change(screen.getByPlaceholderText('Filter by vendor...'), {
      target: { value: 'Zendesk' },
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'battle_card' } })
    fireEvent.change(selects[1], { target: { value: 'sales_ready' } })
    fireEvent.change(selects[2], { target: { value: 'stale' } })
    fireEvent.change(selects[3], { target: { value: 'blocked' } })

    await waitFor(() => {
      expect(api.fetchReports).toHaveBeenLastCalledWith({
        report_type: 'battle_card',
        vendor_filter: 'Zendesk',
        quality_status: 'sales_ready',
        freshness_state: 'stale',
        review_state: 'blocked',
        include_stale: true,
        limit: 100,
      })
    })

    expect(screen.getByRole('button', { name: 'Subscribe to View' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Subscribe to View' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    await waitFor(() => {
      expect(api.buildReportLibraryViewScopeKey).toHaveBeenLastCalledWith({
        report_type: 'battle_card',
        vendor_filter: 'Zendesk',
        quality_status: 'sales_ready',
        freshness_state: 'stale',
        review_state: 'blocked',
      })
    })
    expect(modalState.lastProps.scopeType).toBe('library_view')
    expect(modalState.lastProps.scopeKey).toBe('library-view::battle_card::Zendesk::sales_ready')
    expect(modalState.lastProps.scopeLabel).toBe('Battle Card • Zendesk • Sales Ready • Stale • Blocked Library')
    expect(modalState.lastProps.filterPayload).toEqual({
      report_type: 'battle_card',
      vendor_filter: 'Zendesk',
      quality_status: 'sales_ready',
      freshness_state: 'stale',
      review_state: 'blocked',
    })
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'library_view:library-view::battle_card::Zendesk::sales_ready:Battle Card • Zendesk • Sales Ready • Stale • Blocked Library',
    )
  })

  it('renders API trust states from the report payload', async () => {
    api.fetchReports.mockResolvedValueOnce({
      reports: [
        {
          id: 'report-trust-1',
          report_type: 'vendor_scorecard',
          vendor_filter: 'Intercom',
          category_filter: null,
          executive_summary: 'Trust summary',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'completed',
          artifact_state: 'failed',
          artifact_label: 'Attention needed',
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          freshness_state: 'stale',
          freshness_label: 'Stale',
          review_state: 'blocked',
          review_label: 'Blocked',
          quality_status: null,
          latest_failure_step: 'operator_review',
          latest_error_summary: 'Evidence coverage is below threshold',
          trust: {
            artifact_state: 'failed',
            artifact_label: 'Attention needed',
            freshness_state: 'stale',
            freshness_label: 'Stale',
            review_state: 'blocked',
            review_label: 'Blocked',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText('Trust summary')).toBeInTheDocument()
    const card = screen.getByRole('button', { name: /Trust summary/i })
    expect(within(card).getByText('Attention needed')).toBeInTheDocument()
    expect(within(card).getByText('Blocked')).toBeInTheDocument()
    expect(within(card).getByText('Stale')).toBeInTheDocument()
    expect(within(card).getByText('step: operator review')).toBeInTheDocument()
    expect(within(card).getByText('Evidence coverage is below threshold')).toBeInTheDocument()
  })
})
