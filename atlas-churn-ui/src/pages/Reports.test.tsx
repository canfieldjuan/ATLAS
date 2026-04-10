import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

    expect(screen.getByRole('button', { name: 'Subscribe to View' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Subscribe to View' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    await waitFor(() => {
      expect(api.buildReportLibraryViewScopeKey).toHaveBeenLastCalledWith(expect.objectContaining({
        report_type: 'battle_card',
        vendor_filter: 'Zendesk',
        quality_status: 'sales_ready',
      }))
    })
    expect(modalState.lastProps.scopeType).toBe('library_view')
    expect(modalState.lastProps.scopeKey).toBe('library-view::battle_card::Zendesk::sales_ready')
    expect(modalState.lastProps.scopeLabel).toBe('Battle Card • Zendesk • Sales Ready Library')
    expect(modalState.lastProps.filterPayload).toEqual(expect.objectContaining({
      report_type: 'battle_card',
      vendor_filter: 'Zendesk',
      quality_status: 'sales_ready',
    }))
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'library_view:library-view::battle_card::Zendesk::sales_ready:Battle Card • Zendesk • Sales Ready Library',
    )
  })
})
