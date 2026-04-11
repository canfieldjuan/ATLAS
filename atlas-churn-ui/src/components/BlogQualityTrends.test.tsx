import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import BlogQualityTrends from './BlogQualityTrends'

const trendSpy = vi.hoisted(() => vi.fn())

vi.mock('./CampaignQualityTrends', () => ({
  default: (props: unknown) => {
    trendSpy(props)
    return <div data-testid="campaign-quality-trends" />
  },
}))

describe('BlogQualityTrends', () => {
  it('passes blog trend data with the default title', () => {
    render(
      <BlogQualityTrends
        data={{
          days: 7,
          top_n: 3,
          top_blockers: [],
          series: [],
          totals_by_day: [],
        }}
      />,
    )

    expect(screen.getByTestId('campaign-quality-trends')).toBeInTheDocument()
    expect(trendSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        loading: false,
        title: 'Blog Quality Trends',
        data: expect.objectContaining({
          days: 7,
          top_n: 3,
        }),
      }),
    )
  })

  it('passes explicit loading and title overrides', () => {
    render(<BlogQualityTrends loading title="Editorial Trendline" data={null} />)

    expect(trendSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        loading: true,
        title: 'Editorial Trendline',
        data: null,
      }),
    )
  })
})
