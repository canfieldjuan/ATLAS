import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReportActionBar from './ReportActionBar'

const api = vi.hoisted(() => ({
  downloadReportPdf: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('ReportActionBar', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.useRealTimers()
    window.history.replaceState({}, '', '/reports/report-1')
    Object.defineProperty(navigator, 'clipboard', {
      value: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
      configurable: true,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('downloads the PDF when export is available', async () => {
    const user = userEvent.setup()

    render(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={() => {}}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'PDF' }))
    expect(api.downloadReportPdf).toHaveBeenCalledWith('report-1')
  })

  it('shows disabled PDF states based on artifact status', () => {
    const { rerender } = render(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={() => {}}
        hasPdfExport={false}
        artifactState="processing"
        artifactLabel="Processing"
      />,
    )

    expect(screen.getByRole('button', { name: 'Generating' })).toBeDisabled()

    rerender(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={() => {}}
        hasPdfExport={false}
        artifactState="failed"
        artifactLabel="Failed"
      />,
    )
    expect(screen.getByRole('button', { name: 'PDF Failed' })).toBeDisabled()

    rerender(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={() => {}}
        hasPdfExport={false}
      />,
    )
    expect(screen.getByRole('button', { name: 'Unavailable' })).toBeDisabled()
  })

  it('copies the current report link and resets the copied state', async () => {
    vi.useFakeTimers()

    render(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={() => {}}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Link' }))
    await act(async () => {
      await Promise.resolve()
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('http://localhost:3000/reports/report-1')
    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()

    await act(async () => {
      vi.advanceTimersByTime(2000)
      await Promise.resolve()
    })
    expect(screen.getByRole('button', { name: 'Link' })).toBeInTheDocument()
  })

  it('uses explicit link targets and renders subscription states', async () => {
    vi.useFakeTimers()
    const onSubscribe = vi.fn()

    const { rerender } = render(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={onSubscribe}
        linkUrl="https://atlas.example/custom-report"
        subscriptionState="paused"
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Link' }))
    await act(async () => {
      await Promise.resolve()
    })
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('https://atlas.example/custom-report')
    expect(screen.getByRole('button', { name: 'Resume Subscription' })).toBeInTheDocument()

    rerender(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={onSubscribe}
        hasSubscription
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Manage Subscription' }))
    expect(onSubscribe).toHaveBeenCalledTimes(1)

    rerender(
      <ReportActionBar
        reportId="report-1"
        onSubscribe={onSubscribe}
        subscriptionState="none"
      />,
    )
    expect(screen.getByRole('button', { name: 'Subscribe' })).toBeInTheDocument()
  })
})
