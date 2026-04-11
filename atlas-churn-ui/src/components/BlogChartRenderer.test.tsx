import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BlogChart from './BlogChartRenderer'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div data-testid="responsive">{children}</div>,
  BarChart: ({ children }: { children: ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  LineChart: ({ children }: { children: ReactNode }) => <div data-testid="line-chart">{children}</div>,
  RadarChart: ({ children }: { children: ReactNode }) => <div data-testid="radar-chart">{children}</div>,
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => <div data-testid="legend" />,
  PolarGrid: () => null,
  PolarAngleAxis: () => null,
  PolarRadiusAxis: () => null,
  Bar: ({ dataKey, fill }: { dataKey: string; fill: string }) => <div>{`bar:${dataKey}:${fill}`}</div>,
  Line: ({ dataKey, stroke }: { dataKey: string; stroke: string }) => <div>{`line:${dataKey}:${stroke}`}</div>,
  Radar: ({ dataKey, stroke }: { dataKey: string; stroke: string }) => <div>{`radar:${dataKey}:${stroke}`}</div>,
}))

describe('BlogChartRenderer', () => {
  beforeEach(() => {
    cleanup()
  })

  it('renders the chart title and bar series colors', () => {
    render(
      <BlogChart
        spec={{
          chart_id: 'source-share',
          chart_type: 'bar',
          title: 'Source Share',
          data: [{ name: 'Alpha', wins: 4, losses: 2 }],
          config: {
            bars: [
              { dataKey: 'wins', color: '#123456' },
              { dataKey: 'losses' },
            ],
          },
        }}
      />,
    )

    expect(screen.getByText('Source Share')).toBeInTheDocument()
    expect(screen.getByText('bar:wins:#123456')).toBeInTheDocument()
    expect(screen.getByText('bar:losses:#f472b6')).toBeInTheDocument()
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument()
    expect(screen.getByTestId('legend')).toBeInTheDocument()
  })

  it('renders radar series with fallback colors', () => {
    render(
      <BlogChart
        spec={{
          chart_id: 'churn-radar',
          chart_type: 'radar',
          title: 'Churn Radar',
          data: [{ name: 'Pain', churn: 7 }],
          config: {
            bars: [{ dataKey: 'churn' }],
          },
        }}
      />,
    )

    expect(screen.getByTestId('radar-chart')).toBeInTheDocument()
    expect(screen.getByText('radar:churn:#22d3ee')).toBeInTheDocument()
  })

  it('returns null for unsupported chart types', () => {
    const { container } = render(
      <BlogChart
        spec={{
          chart_id: 'unsupported',
          chart_type: 'scatter',
          title: 'Unsupported',
          data: [],
          config: {},
        } as any}
      />,
    )

    expect(container).toBeEmptyDOMElement()
  })
})
