import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CampaignDiagnostics from './CampaignDiagnostics'

const api = vi.hoisted(() => ({
  fetchCampaignQualityDiagnostics: vi.fn(),
  fetchCampaignQualityTrends: vi.fn(),
}))

vi.mock('../api/client', () => api)

function LocationEcho() {
  const location = useLocation()
  return <div>{`${location.pathname}${location.search}`}</div>
}

const diagnosticsResponse = {
  days: 14,
  by_boundary: [],
  by_cause_type: [],
  top_primary_blockers: [],
  top_missing_inputs: [],
  by_target_mode: [],
  top_vendors: [],
}

const trendsResponse = {
  days: 14,
  top_n: 5,
  top_blockers: [],
  series: [],
  totals_by_day: [],
}

describe('CampaignDiagnostics', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchCampaignQualityDiagnostics.mockResolvedValue(diagnosticsResponse)
    api.fetchCampaignQualityTrends.mockResolvedValue(trendsResponse)
  })

  it('hydrates filters from the URL and loads both campaign diagnostics endpoints', async () => {
    render(
      <MemoryRouter initialEntries={['/campaign-diagnostics?days=30&diagnosticsTopN=20&trendsTopN=10']}>
        <CampaignDiagnostics />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Campaign Diagnostics' })).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchCampaignQualityDiagnostics).toHaveBeenCalledWith({ days: 30, top_n: 20 })
    })
    expect(api.fetchCampaignQualityTrends).toHaveBeenCalledWith({ days: 30, top_n: 10 })

    expect(screen.getByLabelText('Window')).toHaveValue('30')
    expect(screen.getByLabelText('Diagnostics')).toHaveValue('20')
    expect(screen.getByLabelText('Trends')).toHaveValue('10')
    expect(screen.getByRole('link', { name: 'Review Queue' })).toHaveAttribute('href', '/campaign-review')
  })

  it('persists filter changes into the URL and refreshes the campaign diagnostics queries', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/campaign-diagnostics']}>
        <Routes>
          <Route path="/campaign-diagnostics" element={<CampaignDiagnostics />} />
          <Route path="*" element={<LocationEcho />} />
        </Routes>
        <LocationEcho />
      </MemoryRouter>,
    )

    expect(await screen.findByText('/campaign-diagnostics')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchCampaignQualityDiagnostics).toHaveBeenCalledWith({ days: 14, top_n: 10 })
    })
    expect(api.fetchCampaignQualityTrends).toHaveBeenCalledWith({ days: 14, top_n: 5 })

    await user.selectOptions(screen.getByLabelText('Window'), '60')
    await user.selectOptions(screen.getByLabelText('Diagnostics'), '5')
    await user.selectOptions(screen.getByLabelText('Trends'), '20')

    await waitFor(() => {
      expect(screen.getByText('/campaign-diagnostics?days=60&diagnosticsTopN=5&trendsTopN=20')).toBeInTheDocument()
    })
    expect(api.fetchCampaignQualityDiagnostics).toHaveBeenLastCalledWith({ days: 60, top_n: 5 })
    expect(api.fetchCampaignQualityTrends).toHaveBeenLastCalledWith({ days: 60, top_n: 20 })
  })
})
