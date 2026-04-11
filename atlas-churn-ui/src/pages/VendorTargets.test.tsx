import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import VendorTargets from './VendorTargets'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchVendorTargets: vi.fn(),
  createVendorTarget: vi.fn(),
  updateVendorTarget: vi.fn(),
  deleteVendorTarget: vi.fn(),
  generateVendorReport: vi.fn(),
  generateCampaigns: vi.fn(),
  claimVendorTarget: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('VendorTargets', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchVendorTargets.mockResolvedValue({
      targets: [
        {
          id: 'target-1',
          company_name: 'Zendesk',
          target_mode: 'challenger_intel',
          tier: 'dashboard',
          status: 'active',
          ownership_scope: 'account',
          contact_name: 'Alex',
          contact_role: 'VP Support',
          products_tracked: ['Suite'],
          competitors_tracked: ['Freshdesk'],
        },
      ],
    })
    api.createVendorTarget.mockResolvedValue({})
    api.updateVendorTarget.mockResolvedValue({})
    api.deleteVendorTarget.mockResolvedValue({})
    api.generateVendorReport.mockResolvedValue({ signal_count: 3, high_urgency_count: 1 })
    api.generateCampaigns.mockResolvedValue({ generated: 1 })
    api.claimVendorTarget.mockResolvedValue({ already_claimed: false })
  })

  it('hydrates filters from the URL and keeps direct handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/vendor-targets?search=Zendesk&mode=challenger_intel']}>
        <Routes>
          <Route path="/vendor-targets" element={<VendorTargets />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(api.fetchVendorTargets).toHaveBeenCalledWith({
      target_mode: 'challenger_intel',
      search: 'Zendesk',
      limit: 100,
    })

    expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
    )
  })

  it('navigates row clicks to vendor detail with the current list state as back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendor-targets?search=Zendesk&mode=challenger_intel']}>
        <Routes>
          <Route path="/vendor-targets" element={<VendorTargets />} />
        </Routes>
      </MemoryRouter>,
    )

    const vendorCell = await screen.findByText('Zendesk')
    await user.click(vendorCell)

    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
    )
  })

  it('updates the URL-backed handoff links after the applied search filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/vendor-targets']}>
        <Routes>
          <Route path="/vendor-targets" element={<VendorTargets />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search company...'), {
      target: { value: 'Zendesk' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
        'href',
        '/vendors/Zendesk?back_to=%2Fvendor-targets%3Fsearch%3DZendesk',
      )
    })
  })
})
