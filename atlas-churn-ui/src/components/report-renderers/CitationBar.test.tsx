import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import CitationBar from './CitationBar'

describe('CitationBar', () => {
  it('collapses long citation lists until expanded', async () => {
    const user = userEvent.setup()
    const onOpenWitness = vi.fn()
    const citations = Array.from({ length: 6 }, (_, index) => ({
      index: index + 1,
      witnessId: `witness-${index + 1}`,
      companyName: `Company ${index + 1}`,
      excerptSnippet: `Excerpt ${index + 1}`,
    }))

    render(
      <CitationBar
        citations={citations}
        vendorName="Zendesk"
        onOpenWitness={onOpenWitness}
      />,
    )

    expect(screen.getByRole('button', { name: '[1]' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '[5]' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '[6]' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /details/i }))

    expect(screen.getByRole('button', { name: '[6]' })).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: '[6]' })[0])

    expect(onOpenWitness).toHaveBeenCalledWith('witness-6', 'Zendesk')
  })
})
