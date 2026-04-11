import { cleanup, render, screen } from '@testing-library/react'
import { BarChart3 } from 'lucide-react'
import { beforeEach, describe, expect, it } from 'vitest'
import StatCard from './StatCard'

describe('StatCard', () => {
  beforeEach(() => {
    cleanup()
  })

  it('renders the label, value, subtitle, and icon in the loaded state', () => {
    render(
      <StatCard
        label="Tracked Vendors"
        value={42}
        icon={<BarChart3 aria-label="Tracked icon" className="h-4 w-4" />}
        sub="Last 30 days"
      />,
    )

    expect(screen.getByText('Tracked Vendors')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
    expect(screen.getByText('Last 30 days')).toBeInTheDocument()
    expect(screen.getByLabelText('Tracked icon')).toBeInTheDocument()
  })

  it('renders the skeleton shell instead of the live text when loading', () => {
    const { container } = render(
      <StatCard
        label="Tracked Vendors"
        value={42}
        icon={<BarChart3 className="h-4 w-4" />}
        sub="Last 30 days"
        skeleton
      />,
    )

    expect(screen.queryByText('Tracked Vendors')).not.toBeInTheDocument()
    expect(screen.queryByText('42')).not.toBeInTheDocument()
    expect(screen.queryByText('Last 30 days')).not.toBeInTheDocument()
    expect(container.firstChild).toHaveClass('animate-pulse')
  })
})
