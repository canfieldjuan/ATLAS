import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ArchetypeBadge from './ArchetypeBadge'

describe('ArchetypeBadge', () => {
  it('renders nothing when there is no archetype', () => {
    const { container } = render(<ArchetypeBadge archetype={null} />)

    expect(container).toBeEmptyDOMElement()
  })

  it('renders the mapped label, confidence, and known tone for supported archetypes', () => {
    render(
      <ArchetypeBadge
        archetype="pricing_shock"
        confidence={0.84}
        showConfidence
        size="md"
      />,
    )

    const badge = screen.getByText('Pricing Shock').closest('span')
    expect(badge).toHaveClass(
      'bg-amber-500/20',
      'text-amber-300',
      'border-amber-500/30',
      'px-3',
      'py-1',
      'text-sm',
    )
    expect(screen.getByText('84%')).toBeInTheDocument()
  })

  it('falls back to a humanized label and mixed tone for unknown archetypes', () => {
    render(<ArchetypeBadge archetype="renewal_blocker" />)

    const badge = screen.getByText('renewal blocker').closest('span')
    expect(badge).toHaveClass('bg-slate-500/20', 'text-slate-300', 'border-slate-500/30')
  })
})
