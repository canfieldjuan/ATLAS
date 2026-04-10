import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { RouterProvider, createMemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ReportDetail from './ReportDetail'

const modalState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const drawerState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const api = vi.hoisted(() => ({
  fetchReport: vi.fn(),
}))

vi.mock('../api/client', () => ({
  fetchReport: api.fetchReport,
}))
vi.mock('../components/ReportActionBar', () => ({
  default: ({ onSubscribe }: { onSubscribe: () => void }) => (
    <button onClick={onSubscribe}>Open subscription</button>
  ),
}))
vi.mock('../components/SubscriptionModal', () => ({
  default: (props: any) => {
    modalState.lastProps = props
    if (!props.open) return null
    return <div data-testid="subscription-modal">{props.scopeType}:{props.scopeKey}:{props.scopeLabel}</div>
  },
}))
vi.mock('../components/EvidenceDrawer', () => ({
  default: (props: any) => {
    drawerState.lastProps = props
    return props.open ? <div data-testid="evidence-drawer">{props.vendorName}:{props.witnessId}</div> : null
  },
}))

describe('ReportDetail', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    modalState.lastProps = null
    drawerState.lastProps = null
    api.fetchReport.mockResolvedValue({
      id: 'report-1',
      report_type: 'vendor_scorecard',
      vendor_filter: 'Zendesk',
      category_filter: null,
      executive_summary: 'Executive summary',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      llm_model: 'gpt-test',
      status: 'completed',
      blocker_count: 0,
      warning_count: 1,
      unresolved_issue_count: 1,
      quality_status: 'needs_review',
      latest_failure_step: null,
      latest_error_summary: null,
      data_density: {},
      intelligence_data: {
        reasoning_reference_ids: { witness_ids: ['w1'] },
        reasoning_witness_highlights: [
          {
            witness_id: 'w1',
            reviewer_company: 'Acme',
            excerpt_text: 'Pricing changed overnight',
          },
        ],
        key_insights: [
          { label: 'Pricing friction', summary: 'Pricing created churn risk' },
        ],
        key_insights_reference_ids: { witness_ids: ['w1'] },
        key_insights_witness_highlights: [
          {
            witness_id: 'w1',
            reviewer_company: 'Acme',
            excerpt_text: 'Pricing created churn risk',
          },
        ],
      },
    })
  })

  it('opens the report subscription modal with the report scope metadata', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-1'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Zendesk')
    fireEvent.click(screen.getByText('Open subscription'))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-1')
    expect(modalState.lastProps.scopeLabel).toBe('vendor scorecard - Zendesk')
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'report:report-1:vendor scorecard - Zendesk',
    )
  })

  it('opens the evidence drawer from executive summary citations', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-1'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Executive summary')
    fireEvent.click(screen.getAllByRole('button', { name: '[1]' })[0])

    await waitFor(() => {
      expect(drawerState.lastProps?.open).toBe(true)
    })

    expect(drawerState.lastProps.vendorName).toBe('Zendesk')
    expect(drawerState.lastProps.witnessId).toBe('w1')
    expect(screen.getByTestId('evidence-drawer')).toHaveTextContent('Zendesk:w1')
  })
})
