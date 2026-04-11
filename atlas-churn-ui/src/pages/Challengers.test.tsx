import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Challengers from './Challengers'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchVendorTargets: vi.fn(),
  fetchHighIntent: vi.fn(),
  generateCampaigns: vi.fn(),
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

describe('Challengers', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchVendorTargets.mockResolvedValue({
      targets: [
        {
          id: 'target-1',
          company_name: 'Zendesk',
          competitors_tracked: ['Freshdesk'],
        },
      ],
    })
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          id: 'company-1',
          company_name: 'Acme Corp',
          vendor: 'Freshdesk',
          alternatives: [{ name: 'Zendesk' }],
          buying_stage: 'active_purchase',
          pain: 'support',
          urgency: 8,
        },
      ],
    })
    api.generateCampaigns.mockResolvedValue({ generated: 1 })
  })

  it('hydrates search from the URL and keeps direct handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/challengers?search=Zendesk']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(api.fetchVendorTargets).toHaveBeenCalledWith({ target_mode: 'challenger_intel', limit: 100 })
    expect(api.fetchHighIntent).toHaveBeenCalledWith({ min_urgency: 3, limit: 100 })

    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
  })

  it('navigates row clicks to vendor detail with the current list state as back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/challengers?search=Zendesk']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    const vendorCell = await screen.findByText('Zendesk')
    await user.click(vendorCell)

    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
  })

  it('updates the URL-backed handoff links after the applied search filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search challenger...'), {
      target: { value: 'Zendesk' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
        'href',
        '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fchallengers%3Fsearch%3DZendesk',
      )
    })
  })
})
