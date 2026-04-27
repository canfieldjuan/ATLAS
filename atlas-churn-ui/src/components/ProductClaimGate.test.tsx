import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { VendorClaim } from '../api/client'
import {
  ProductClaimGate,
  ProductClaimStatusBadge,
  isProductClaimAllowed,
  productClaimGateState,
} from './ProductClaimGate'

function claim(overrides: Partial<VendorClaim> = {}): VendorClaim {
  return {
    claim_id: 'claim-1',
    claim_key: 'claim-key',
    claim_scope: 'competitor_pair',
    claim_type: 'direct_displacement',
    claim_text: 'Validated product claim',
    target_entity: 'Microsoft 365',
    secondary_target: 'Google Workspace',
    supporting_count: 3,
    direct_evidence_count: 3,
    witness_count: 2,
    contradiction_count: 0,
    denominator: null,
    sample_size: 3,
    has_grounded_evidence: true,
    confidence: 'medium',
    evidence_posture: 'usable',
    render_allowed: true,
    report_allowed: false,
    suppression_reason: 'low_confidence',
    evidence_links: ['review-1'],
    contradicting_links: [],
    as_of_date: '2026-04-26',
    analysis_window_days: 30,
    schema_version: 'product_claim.v1',
    ...overrides,
  }
}

describe('ProductClaimGate', () => {
  afterEach(() => cleanup())

  it('fails closed when validation is unavailable', () => {
    expect(isProductClaimAllowed(undefined, 'render', true)).toBe(false)
    expect(isProductClaimAllowed(undefined, 'report', true)).toBe(false)
    expect(productClaimGateState(undefined, true)).toBe('validation_unavailable')

    render(<ProductClaimStatusBadge claim={undefined} validationUnavailable />)

    expect(screen.getByText('Validation unavailable')).toBeInTheDocument()
  })

  it('treats missing claims as legacy state but blocks render and report gates', () => {
    expect(productClaimGateState(undefined)).toBe('legacy')
    expect(isProductClaimAllowed(undefined, 'render')).toBe(false)
    expect(isProductClaimAllowed(undefined, 'report')).toBe(false)

    render(
      <ProductClaimGate claim={undefined} mode="render" testId="gate">
        Unsafe value
      </ProductClaimGate>,
    )

    expect(screen.queryByText('Unsafe value')).not.toBeInTheDocument()
    expect(screen.getByTestId('gate')).toHaveTextContent('Legacy fallback')
  })

  it('blocks render content when render_allowed is false', () => {
    const suppressed = claim({
      render_allowed: false,
      report_allowed: false,
      suppression_reason: 'unverified_evidence',
    })

    render(
      <ProductClaimGate claim={suppressed} mode="render" testId="gate">
        Winner call
      </ProductClaimGate>,
    )

    expect(productClaimGateState(suppressed)).toBe('insufficient')
    expect(screen.queryByText('Winner call')).not.toBeInTheDocument()
    expect(screen.getByTestId('gate')).toHaveTextContent('Insufficient')
  })

  it('allows display but blocks report actions for monitor-only rows', () => {
    const monitor = claim()

    expect(productClaimGateState(monitor)).toBe('monitor')
    expect(isProductClaimAllowed(monitor, 'render')).toBe(true)
    expect(isProductClaimAllowed(monitor, 'report')).toBe(false)

    render(
      <ProductClaimGate claim={monitor} mode="report" testId="gate">
        Generate
      </ProductClaimGate>,
    )

    expect(screen.queryByText('Generate')).not.toBeInTheDocument()
    expect(screen.getByTestId('gate')).toHaveTextContent('Monitor only')
  })

  it('allows report-safe rows through both gates', () => {
    const safe = claim({
      report_allowed: true,
      suppression_reason: null,
      confidence: 'medium',
    })

    expect(productClaimGateState(safe)).toBe('report_safe')
    expect(isProductClaimAllowed(safe, 'render')).toBe(true)
    expect(isProductClaimAllowed(safe, 'report')).toBe(true)

    render(
      <ProductClaimGate claim={safe} mode="report">
        Generate
      </ProductClaimGate>,
    )

    expect(screen.getByText('Generate')).toBeInTheDocument()
  })
})
