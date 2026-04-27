import type { ReactNode } from 'react'
import type { VendorClaim } from '../api/client'
import { suppressionLabel } from '../lib/suppressionLabels'

export type ProductClaimGateState =
  | 'validation_unavailable'
  | 'legacy'
  | 'insufficient'
  | 'monitor'
  | 'report_safe'

export type ProductClaimGateMode = 'render' | 'report'

export function productClaimGateState(
  claim: VendorClaim | null | undefined,
  validationUnavailable = false,
): ProductClaimGateState {
  if (validationUnavailable) return 'validation_unavailable'
  if (!claim) return 'legacy'
  if (claim.render_allowed !== true) return 'insufficient'
  if (claim.report_allowed !== true) return 'monitor'
  return 'report_safe'
}

export function isProductClaimAllowed(
  claim: VendorClaim | null | undefined,
  mode: ProductClaimGateMode,
  validationUnavailable = false,
): boolean {
  if (validationUnavailable) return false
  if (!claim) return false
  return mode === 'report'
    ? claim.report_allowed === true
    : claim.render_allowed === true
}

export function productClaimGateTitle(
  claim: VendorClaim | null | undefined,
  validationUnavailable = false,
): string {
  if (validationUnavailable) return 'Validation unavailable: claim service did not return a usable response'
  if (!claim) return 'Legacy row: validation claim unavailable'
  const reason = suppressionLabel(claim.suppression_reason)
  const countText = `${claim.supporting_count} supporting / ${claim.direct_evidence_count} direct / ${claim.witness_count} witnesses`
  if (claim.render_allowed !== true) {
    return `${reason ?? 'Insufficient evidence'} (${countText})`
  }
  if (claim.report_allowed !== true) {
    return `${reason ?? 'Monitor only'} (${countText})`
  }
  return `Report-safe (${countText})`
}

export function ProductClaimStatusBadge({
  claim,
  validationUnavailable = false,
}: {
  claim: VendorClaim | null | undefined
  validationUnavailable?: boolean
}) {
  const state = productClaimGateState(claim, validationUnavailable)
  if (state === 'validation_unavailable') {
    return (
      <span
        className="inline-flex items-center rounded-full bg-slate-700/30 px-2 py-0.5 text-[10px] font-medium text-slate-400"
        title={productClaimGateTitle(claim, validationUnavailable)}
      >
        Validation unavailable
      </span>
    )
  }
  if (state === 'legacy') {
    return (
      <span
        className="inline-flex items-center rounded-full bg-slate-700/30 px-2 py-0.5 text-[10px] font-medium text-slate-400"
        title={productClaimGateTitle(claim)}
      >
        Legacy
      </span>
    )
  }
  if (state === 'insufficient') {
    return (
      <span
        className="inline-flex items-center rounded-full bg-rose-500/15 px-2 py-0.5 text-[10px] font-medium text-rose-300"
        title={productClaimGateTitle(claim)}
      >
        Insufficient
      </span>
    )
  }
  if (state === 'monitor') {
    return (
      <span
        className="inline-flex items-center rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium text-amber-300"
        title={productClaimGateTitle(claim)}
      >
        Monitor only
      </span>
    )
  }
  return (
    <span
      className="inline-flex items-center rounded-full bg-green-500/15 px-2 py-0.5 text-[10px] font-medium text-green-300"
      title={productClaimGateTitle(claim)}
    >
      Report-safe
    </span>
  )
}

export function ProductClaimGate({
  claim,
  children,
  fallback,
  mode,
  testId,
  validationUnavailable = false,
}: {
  claim: VendorClaim | null | undefined
  children: ReactNode
  fallback?: ReactNode
  mode: ProductClaimGateMode
  testId?: string
  validationUnavailable?: boolean
}) {
  if (isProductClaimAllowed(claim, mode, validationUnavailable)) return <>{children}</>
  const state = productClaimGateState(claim, validationUnavailable)
  const fallbackLabel = fallback ?? (
    state === 'validation_unavailable'
      ? 'Validation unavailable'
      : state === 'legacy'
        ? 'Legacy fallback'
        : state === 'insufficient'
          ? 'Insufficient'
          : 'Monitor only'
  )
  return (
    <span
      className="text-xs text-slate-500 italic"
      title={productClaimGateTitle(claim, validationUnavailable)}
      data-testid={testId}
    >
      {fallbackLabel}
    </span>
  )
}
