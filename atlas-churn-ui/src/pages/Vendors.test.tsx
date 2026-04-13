import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, RouterProvider, createMemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Vendors from './Vendors'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchSignals: vi.fn(),
  downloadCsv: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

describe('Vendors', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchSignals.mockResolvedValue({
      signals: [
        {
          vendor_name: 'Zendesk',
          product_category: 'Helpdesk',
          total_reviews: 20,
          churn_intent_count: 5,
          avg_urgency_score: 7.2,
          avg_rating_normalized: 42,
          nps_proxy: -12.4,
          price_complaint_rate: 0.25,
          decision_maker_churn_rate: 0.4,
          support_sentiment: -1.2,
          legacy_support_score: 3.8,
          new_feature_velocity: 2.5,
          employee_growth_rate: 4.2,
          archetype: 'slow_burn',
          archetype_confidence: 0.84,
          reasoning_mode: null,
          last_computed_at: null,
        },
      ],
    })
  })

  it('hydrates filters from the URL and keeps direct vendor handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/vendors?search=Zendesk&min_urgency=6&category=Helpdesk']}>
        <Routes>
          <Route path="/vendors" element={<Vendors />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchSignals).toHaveBeenLastCalledWith({
        vendor_name: 'Zendesk',
        min_urgency: 6,
        category: 'Helpdesk',
        limit: 100,
      })
    })

    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6%26category%3DHelpdesk',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6%26category%3DHelpdesk',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6%26category%3DHelpdesk',
    )
  })

  it('navigates row clicks to vendor detail with the current list state as back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors?search=Zendesk&min_urgency=6&category=Helpdesk']}>
        <Routes>
          <Route path="/vendors" element={<Vendors />} />
        </Routes>
      </MemoryRouter>,
    )

    const vendorCell = await screen.findByText('Zendesk')
    await user.click(vendorCell)

    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6%26category%3DHelpdesk',
    )
  })

  it('updates the URL-backed handoff links after the applied search filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/vendors']}>
        <Routes>
          <Route path="/vendors" element={<Vendors />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search vendors...'), {
      target: { value: 'Zendesk' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
        'href',
        '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendors%3Fsearch%3DZendesk',
      )
    })
  })

  it('clears same-route list filters without restoring stale query params', async () => {
    const router = createMemoryRouter(
      [{ path: '/vendors', element: <><Vendors /><LocationProbe /></> }],
      {
        initialEntries: ['/vendors?search=Zendesk&min_urgency=6&category=Helpdesk'],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()

    await router.navigate('/vendors')

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search vendors...')).toHaveValue('')
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/vendors')
    })

    await waitFor(() => {
      expect(api.fetchSignals).toHaveBeenLastCalledWith({
        vendor_name: undefined,
        min_urgency: undefined,
        category: undefined,
        limit: 100,
      })
    })
  })

  it('canonicalizes invalid route filters on load', async () => {
    const router = createMemoryRouter(
      [{ path: '/vendors', element: <><Vendors /><LocationProbe /></> }],
      {
        initialEntries: ['/vendors?search=%20Zendesk%20&min_urgency=99&category=%20Helpdesk%20'],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchSignals).toHaveBeenLastCalledWith({
        vendor_name: 'Zendesk',
        min_urgency: 10,
        category: 'Helpdesk',
        limit: 100,
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/vendors?search=Zendesk&min_urgency=10&category=Helpdesk')
    })
  })
})
