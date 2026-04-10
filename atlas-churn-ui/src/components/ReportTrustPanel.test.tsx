import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReportTrustPanel from './ReportTrustPanel'

describe('ReportTrustPanel', () => {
  it('renders clean ready trust states', () => {
    render(
      <ReportTrustPanel
        status="completed"
        blockerCount={0}
        warningCount={0}
        unresolvedIssueCount={0}
        qualityStatus="sales_ready"
        freshnessState="fresh"
        freshnessLabel="Fresh"
        reviewState="clean"
        reviewLabel="Clean"
      />,
    )

    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(screen.getByText('Clean')).toBeInTheDocument()
    expect(screen.getByText('Fresh')).toBeInTheDocument()
    expect(screen.getByText('Sales Ready')).toBeInTheDocument()
  })

  it('renders failure and review warnings', () => {
    render(
      <ReportTrustPanel
        status="failed"
        blockerCount={2}
        warningCount={1}
        unresolvedIssueCount={3}
        latestFailureStep="artifact_generation"
        latestErrorSummary="Missing evidence coverage"
        freshnessState="stale"
        freshnessLabel="Stale"
        reviewState="blocked"
        reviewLabel="Blocked"
      />,
    )

    expect(screen.getByText('Attention needed')).toBeInTheDocument()
    expect(screen.getByText('Blocked')).toBeInTheDocument()
    expect(screen.getByText('Stale')).toBeInTheDocument()
    expect(screen.getByText('2 blockers • 1 warning • 3 open issues')).toBeInTheDocument()
    expect(screen.getByText('step: artifact generation')).toBeInTheDocument()
    expect(screen.getByText('Missing evidence coverage')).toBeInTheDocument()
  })
})
