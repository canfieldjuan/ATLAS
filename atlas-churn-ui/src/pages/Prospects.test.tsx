import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, RouterProvider, Routes, createMemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ProspectsPage from './Prospects'

const api = vi.hoisted(() => ({
  fetchProspects: vi.fn(),
  fetchProspectStats: vi.fn(),
  downloadProspectsCsv: vi.fn(),
  setSequenceRecipient: vi.fn(),
  generateCampaigns: vi.fn(),
  fetchManualQueue: vi.fn(),
  resolveManualQueueEntry: vi.fn(),
  fetchCompanyOverrides: vi.fn(),
  upsertCompanyOverride: vi.fn(),
  deleteCompanyOverride: vi.fn(),
  bootstrapCompanyOverrides: vi.fn(),
}))

vi.mock('../api/client', () => api)

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

describe('ProspectsPage', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchProspectStats.mockResolvedValue({
      total: 1,
      active: 1,
      contacted: 0,
      this_month: 1,
    })
    api.fetchProspects.mockResolvedValue({
      count: 1,
      prospects: [
        {
          id: 'prospect-1',
          company_name: 'Acme Corp',
          company_domain: 'acme.com',
          first_name: 'Alex',
          last_name: 'Stone',
          title: 'VP Support',
          city: 'Chicago',
          state: 'IL',
          email: 'alex@acme.com',
          linkedin_url: null,
          seniority: 'vp',
          status: 'active',
          related_sequence_id: null,
          related_sequence_status: null,
          related_sequence_current_step: null,
          related_sequence_max_steps: null,
          related_sequence_last_sent_at: null,
          churning_from: 'Zendesk',
          target_persona: 'economic_buyer',
          reasoning_atom_context: {
            account_signals: [
              {
                buying_stage: 'evaluation',
                primary_pain: 'support',
                competitor_context: 'Freshdesk',
                decision_timeline: 'Q3',
                quote: 'Support quality is slipping',
              },
            ],
          },
          email_status: 'valid',
          created_at: '2026-04-11T00:00:00Z',
        },
      ],
    })
    api.fetchManualQueue.mockResolvedValue({ count: 0, queue: [] })
    api.fetchCompanyOverrides.mockResolvedValue({ count: 0, overrides: [] })
    api.setSequenceRecipient.mockResolvedValue({})
    api.generateCampaigns.mockResolvedValue({ generated: 1 })
    api.resolveManualQueueEntry.mockResolvedValue({})
    api.upsertCompanyOverride.mockResolvedValue({})
    api.deleteCompanyOverride.mockResolvedValue({})
    api.bootstrapCompanyOverrides.mockResolvedValue({ imported: 0 })
  })

  it('hydrates prospect filters from the URL and keeps vendor handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/prospects?company=Acme&status=active&seniority=vp']}>
        <Routes>
          <Route path="/prospects" element={<ProspectsPage />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Acme')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchProspects).toHaveBeenLastCalledWith({
        company: 'Acme',
        status: 'active',
        seniority: 'vp',
        limit: 200,
      })
    })

    expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
    )
  })

  it('updates the URL-backed handoff links after the applied search filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/prospects']}>
        <Routes>
          <Route path="/prospects" element={<ProspectsPage />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search company...'), {
      target: { value: 'Acme' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
        'href',
        '/vendors/Zendesk?back_to=%2Fprospects%3Fcompany%3DAcme',
      )
    })
  })

  it('clears same-route filters without restoring stale query params', async () => {
    const router = createMemoryRouter(
      [{ path: '/prospects', element: <><ProspectsPage /><LocationProbe /></> }],
      { initialEntries: ['/prospects?company=Acme&status=active&seniority=vp'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Acme')).toBeInTheDocument()

    await router.navigate('/prospects')

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search company...')).toHaveValue('')
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/prospects')
    })

    await waitFor(() => {
      expect(api.fetchProspects).toHaveBeenLastCalledWith({
        company: undefined,
        status: undefined,
        seniority: undefined,
        limit: 200,
      })
    })
  })

  it('rehydrates the active tab from same-route URL changes', async () => {
    const router = createMemoryRouter(
      [{ path: '/prospects', element: <><ProspectsPage /><LocationProbe /></> }],
      { initialEntries: ['/prospects?company=Acme&status=active&seniority=vp'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Acme')).toBeInTheDocument()

    await router.navigate('/prospects?tab=manual_queue')

    await waitFor(() => {
      expect(screen.getByText('0 queue entries')).toBeInTheDocument()
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/prospects?tab=manual_queue')
    })
  })

  it('canonicalizes invalid route filters on load', async () => {
    const router = createMemoryRouter(
      [{ path: '/prospects', element: <><ProspectsPage /><LocationProbe /></> }],
      { initialEntries: ['/prospects?tab=bogus&company=%20Acme%20&status=wat&seniority=king'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Acme')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchProspects).toHaveBeenLastCalledWith({
        company: 'Acme',
        status: undefined,
        seniority: undefined,
        limit: 200,
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/prospects?company=Acme')
    })
  })
})
