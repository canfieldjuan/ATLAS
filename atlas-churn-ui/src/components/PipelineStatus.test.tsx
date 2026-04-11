import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PipelineStatusWidget from './PipelineStatus'
import type { PipelineStatus } from '../types'

describe('PipelineStatusWidget', () => {
  beforeEach(() => {
    cleanup()
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-11T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders the loading skeleton when no pipeline data is available', () => {
    const { container } = render(<PipelineStatusWidget data={null} />)

    expect(container.firstChild).toHaveClass('animate-pulse')
    expect(screen.queryByText('Pipeline Health')).not.toBeInTheDocument()
  })

  it('renders the computed enrichment rate and pipeline stats', () => {
    const data: PipelineStatus = {
      enrichment_counts: {
        enriched: 80,
        pending: 20,
        failed: 10,
      },
      recent_imports_24h: 14,
      last_enrichment_at: '2026-04-11T10:30:00Z',
      active_scrape_targets: 37,
      last_scrape_at: '2026-04-11T09:00:00Z',
    }

    render(<PipelineStatusWidget data={data} />)

    expect(screen.getByText('Pipeline Health')).toBeInTheDocument()
    expect(screen.getByText('73%')).toBeInTheDocument()
    expect(screen.getByText('Enriched: 80')).toBeInTheDocument()
    expect(screen.getByText('Pending: 20')).toBeInTheDocument()
    expect(screen.getByText('1h ago')).toBeInTheDocument()
    expect(screen.getByText('37')).toBeInTheDocument()
    expect(screen.getByText('14')).toBeInTheDocument()
  })

  it('handles empty counts and missing timestamps', () => {
    const data: PipelineStatus = {
      enrichment_counts: {},
      recent_imports_24h: 0,
      last_enrichment_at: null,
      active_scrape_targets: 0,
      last_scrape_at: null,
    }

    render(<PipelineStatusWidget data={data} />)

    expect(screen.getByText('0%')).toBeInTheDocument()
    expect(screen.getByText('Enriched: 0')).toBeInTheDocument()
    expect(screen.getByText('Pending: 0')).toBeInTheDocument()
    expect(screen.getByText('Never')).toBeInTheDocument()
  })
})
