import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import DiagnosticsQueryControls from './DiagnosticsQueryControls'

describe('DiagnosticsQueryControls', () => {
  it('renders current values and forwards select changes', () => {
    const onDaysChange = vi.fn()
    const onDiagnosticsTopNChange = vi.fn()
    const onTrendsTopNChange = vi.fn()

    render(
      <DiagnosticsQueryControls
        days={14}
        diagnosticsTopN={10}
        trendsTopN={5}
        onDaysChange={onDaysChange}
        onDiagnosticsTopNChange={onDiagnosticsTopNChange}
        onTrendsTopNChange={onTrendsTopNChange}
      />,
    )

    expect(screen.getByText('Window')).toBeInTheDocument()
    expect(screen.getByText('Diagnostics')).toBeInTheDocument()
    expect(screen.getByText('Trends')).toBeInTheDocument()

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: '30' } })
    fireEvent.change(selects[1], { target: { value: '20' } })
    fireEvent.change(selects[2], { target: { value: '10' } })

    expect(onDaysChange).toHaveBeenCalledWith(30)
    expect(onDiagnosticsTopNChange).toHaveBeenCalledWith(20)
    expect(onTrendsTopNChange).toHaveBeenCalledWith(10)
  })
})
