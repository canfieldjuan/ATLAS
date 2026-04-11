import { cleanup, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import CompanyTimeline from './CompanyTimeline'

const useApiData = vi.hoisted(() => vi.fn())

vi.mock('../hooks/useApiData', () => ({
  default: useApiData,
}))

describe('CompanyTimeline', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows a loading state while the timeline is fetching', () => {
    useApiData.mockReturnValue({
      data: null,
      loading: true,
      error: null,
      refresh: vi.fn(),
      refreshing: false,
    })

    render(<CompanyTimeline company="Acme Corp" vendor="Zendesk" />)

    expect(screen.getByText('Loading timeline...')).toBeInTheDocument()
  })

  it('shows the API error when the timeline request fails', () => {
    useApiData.mockReturnValue({
      data: null,
      loading: false,
      error: new Error('Timeline request failed'),
      refresh: vi.fn(),
      refreshing: false,
    })

    render(<CompanyTimeline company="Acme Corp" vendor="Zendesk" />)

    expect(screen.getByText('Timeline request failed')).toBeInTheDocument()
  })

  it('shows the empty state when no campaign activity exists', () => {
    useApiData.mockReturnValue({
      data: { events: [] },
      loading: false,
      error: null,
      refresh: vi.fn(),
      refreshing: false,
    })

    render(<CompanyTimeline company="Acme Corp" vendor="Zendesk" />)

    expect(screen.getByText('No campaign activity yet for this company.')).toBeInTheDocument()
  })

  it('renders sequence, signal, and campaign event activity', () => {
    useApiData.mockReturnValue({
      data: {
        events: [
          {
            type: 'sequence_state',
            status: 'completed',
            recipient: 'taylor@acme.com',
            step: 3,
            max_steps: 4,
            open_count: 2,
            click_count: 1,
            outcome: 'deal_won',
            outcome_revenue: 120000,
          },
          {
            type: 'signal_detected',
            timestamp: '2026-04-10T10:30:00Z',
            buying_stage: 'evaluation',
            role_type: 'executive_buyer',
            urgency_score: 9,
            seat_count: 120,
          },
          {
            type: 'campaign_event',
            event: 'opened',
            timestamp: '2026-04-10T11:00:00Z',
            subject: 'Switch from Zendesk before renewal',
            recipient: 'taylor@acme.com',
          },
          {
            type: 'campaign_event',
            event: 'send_failed',
            timestamp: '2026-04-10T12:00:00Z',
            error: 'SMTP timeout',
            recipient: 'taylor@acme.com',
          },
        ],
      },
      loading: false,
      error: null,
      refresh: vi.fn(),
      refreshing: false,
    })

    render(<CompanyTimeline company="Acme Corp" vendor="Zendesk" />)

    expect(screen.getByText('Sequence (completed)')).toBeInTheDocument()
    expect(screen.getByText('deal won')).toBeInTheDocument()
    expect(screen.getByText('$120,000')).toBeInTheDocument()
    expect(screen.getByText('Signal detected')).toBeInTheDocument()
    expect(screen.getByText('evaluation')).toBeInTheDocument()
    expect(screen.getByText('urgency 9')).toBeInTheDocument()
    expect(screen.getByText('120 seats')).toBeInTheDocument()
    expect(screen.getByText('Opened')).toBeInTheDocument()
    expect(screen.getByText('Switch from Zendesk before renewal')).toBeInTheDocument()
    expect(screen.getByText('Send failed')).toBeInTheDocument()
    expect(screen.getByText('SMTP timeout')).toBeInTheDocument()
  })
})
