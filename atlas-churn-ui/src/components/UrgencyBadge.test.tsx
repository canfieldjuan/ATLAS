import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import UrgencyBadge from './UrgencyBadge'

describe('UrgencyBadge', () => {
  it('renders a placeholder when no score is available', () => {
    render(<UrgencyBadge score={null} />)

    expect(screen.getByText('--')).toHaveClass('text-slate-500')
  })

  it('rounds scores and applies the expected tone thresholds', () => {
    const { rerender } = render(<UrgencyBadge score={8.04} />)

    expect(screen.getByText('8')).toHaveClass('bg-red-500/20', 'text-red-400')

    rerender(<UrgencyBadge score={6.2} />)
    expect(screen.getByText('6.2')).toHaveClass('bg-amber-500/20', 'text-amber-400')

    rerender(<UrgencyBadge score={4.4} />)
    expect(screen.getByText('4.4')).toHaveClass('bg-yellow-500/20', 'text-yellow-300')

    rerender(<UrgencyBadge score={3.26} />)
    expect(screen.getByText('3.3')).toHaveClass('bg-green-500/20', 'text-green-400')
  })
})
