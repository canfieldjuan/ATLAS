import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import CitationMarker from './CitationMarker'

describe('CitationMarker', () => {
  it('opens witness detail and stops parent click propagation', async () => {
    const user = userEvent.setup()
    const onParentClick = vi.fn()
    const onOpenWitness = vi.fn()

    render(
      <div onClick={onParentClick}>
        <CitationMarker
          index={3}
          witnessId="witness-3"
          vendorName="Zendesk"
          companyHint="Acme Corp"
          onOpenWitness={onOpenWitness}
        />
      </div>,
    )

    const button = screen.getByRole('button', { name: '[3]' })
    expect(button.getAttribute('title')).toContain('Acme Corp')
    expect(button.getAttribute('title')).toContain('click to view evidence')

    await user.click(button)

    expect(onOpenWitness).toHaveBeenCalledWith('witness-3', 'Zendesk')
    expect(onParentClick).not.toHaveBeenCalled()
  })
})
