import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import BlogDiagnostics from './BlogDiagnostics'

const api = vi.hoisted(() => ({
  fetchBlogQualityDiagnostics: vi.fn(),
  fetchBlogQualityTrends: vi.fn(),
}))

vi.mock('../api/client', () => api)

function LocationEcho() {
  const location = useLocation()
  return <div>{`${location.pathname}${location.search}`}</div>
}

const diagnosticsResponse = {
  days: 14,
  active_failure_count: 0,
  rejected_failure_count: 0,
  current_blocked_slug_count: 0,
  retry_limit_blocked_slug_count: 0,
  by_status: [],
  by_boundary: [],
  by_cause_type: [],
  top_primary_blockers: [],
  top_missing_inputs: [],
  by_topic_type: [],
  top_subjects: [],
  top_blocked_slugs: [],
}

const trendsResponse = {
  days: 14,
  top_n: 5,
  top_blockers: [],
  series: [],
  totals_by_day: [],
}

describe('BlogDiagnostics', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchBlogQualityDiagnostics.mockResolvedValue(diagnosticsResponse)
    api.fetchBlogQualityTrends.mockResolvedValue(trendsResponse)
  })

  it('hydrates filters from the URL and loads both blog diagnostics endpoints', async () => {
    render(
      <MemoryRouter initialEntries={['/blog-diagnostics?days=30&diagnosticsTopN=20&trendsTopN=10']}>
        <BlogDiagnostics />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Blog Diagnostics' })).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchBlogQualityDiagnostics).toHaveBeenCalledWith({ days: 30, top_n: 20 })
    })
    expect(api.fetchBlogQualityTrends).toHaveBeenCalledWith({ days: 30, top_n: 10 })

    expect(screen.getByLabelText('Window')).toHaveValue('30')
    expect(screen.getByLabelText('Diagnostics')).toHaveValue('20')
    expect(screen.getByLabelText('Trends')).toHaveValue('10')
    expect(screen.getByRole('link', { name: 'Blog Review' })).toHaveAttribute('href', '/blog-review')
  })

  it('persists filter changes into the URL and refreshes the blog diagnostics queries', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/blog-diagnostics']}>
        <Routes>
          <Route path="/blog-diagnostics" element={<BlogDiagnostics />} />
          <Route path="*" element={<LocationEcho />} />
        </Routes>
        <LocationEcho />
      </MemoryRouter>,
    )

    expect(await screen.findByText('/blog-diagnostics')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchBlogQualityDiagnostics).toHaveBeenCalledWith({ days: 14, top_n: 10 })
    })
    expect(api.fetchBlogQualityTrends).toHaveBeenCalledWith({ days: 14, top_n: 5 })

    await user.selectOptions(screen.getByLabelText('Window'), '60')
    await user.selectOptions(screen.getByLabelText('Diagnostics'), '5')
    await user.selectOptions(screen.getByLabelText('Trends'), '20')

    await waitFor(() => {
      expect(screen.getByText('/blog-diagnostics?days=60&diagnosticsTopN=5&trendsTopN=20')).toBeInTheDocument()
    })
    expect(api.fetchBlogQualityDiagnostics).toHaveBeenLastCalledWith({ days: 60, top_n: 5 })
    expect(api.fetchBlogQualityTrends).toHaveBeenLastCalledWith({ days: 60, top_n: 20 })
  })
})
