import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Dashboard from './Dashboard'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchSignals: vi.fn(),
  fetchSlowBurnWatchlist: vi.fn(),
  fetchHighIntent: vi.fn(),
  fetchPipeline: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('../components/ChurnChart', () => ({
  default: () => <div>Churn Chart</div>,
}))

vi.mock('../components/PipelineStatus', () => ({
  default: () => <div>Pipeline Status</div>,
}))

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Dashboard', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchSignals.mockResolvedValue({
      signals: [
        {
          vendor_name: 'Zendesk',
          product_category: 'Helpdesk',
          avg_urgency_score: 7.8,
          support_sentiment: -1.1,
          legacy_support_score: 4.1,
          new_feature_velocity: 2.3,
          employee_growth_rate: 6.4,
          archetype: 'slow_burn',
          archetype_confidence: 0.82,
          total_reviews: 20,
          last_computed_at: '2026-04-11T00:00:00Z',
        },
      ],
      total_vendors: 1,
      high_urgency_count: 1,
      total_signal_reviews: 20,
    })
    api.fetchSlowBurnWatchlist.mockResolvedValue({
      signals: [
        {
          vendor_name: 'Zendesk',
          product_category: 'Helpdesk',
          support_sentiment: -1.1,
          legacy_support_score: 4.1,
          new_feature_velocity: 2.3,
          employee_growth_rate: 6.4,
          archetype: 'slow_burn',
          archetype_confidence: 0.82,
        },
      ],
    })
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.2,
          pain: 'support',
          seat_count: 120,
          buying_stage: 'evaluation',
          contract_end: '2026-09-01',
          decision_maker: true,
        },
      ],
    })
    api.fetchPipeline.mockResolvedValue({
      enrichment_counts: { enriched: 12, pending: 3 },
      last_enrichment_at: '2026-04-11T00:00:00Z',
      last_scrape_at: '2026-04-10T00:00:00Z',
    })
  })

  it('surfaces vendor, evidence, report, and opportunity handoffs from the high-intent table', async () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Churn Signals Overview' })).toBeInTheDocument()
    const companyRow = screen.getByText('Acme Corp').closest('tr')
    expect(companyRow).not.toBeNull()
    const row = within(companyRow as HTMLTableRowElement)
    expect(row.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fdashboard',
    )
    expect(row.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fdashboard',
    )
    expect(row.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fdashboard',
    )
    expect(row.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fdashboard',
    )
  })

  it('navigates slow-burn cards and company rows into vendor detail with dashboard back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('button', { name: /Zendesk Helpdesk/i })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Zendesk Helpdesk/i }))
    expect(mockNavigate).toHaveBeenCalledWith('/vendors/Zendesk?back_to=%2Fdashboard')

    await user.click(screen.getByText('Acme Corp'))
    expect(mockNavigate).toHaveBeenCalledWith('/vendors/Zendesk?back_to=%2Fdashboard')
  })
})
