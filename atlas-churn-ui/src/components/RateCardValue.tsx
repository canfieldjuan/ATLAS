/**
 * Phase 10 Patch 2c. Shared render for VENDOR-scope rate cards on the
 * dashboard. Distinguishes the four states the contract specifies:
 *
 *   1. validationUnavailable -- the claims API call failed
 *      (claimsResponse === null). Renders 'Validation unavailable';
 *      legacy rate is NOT shown because we cannot prove it is safe.
 *   2. claim present + render_allowed=true + denominator > 0 ->
 *      computed rate from claim.supporting_count / claim.denominator.
 *      Legacy rate is NOT shown; the contract is the source of truth.
 *   3. claim present + render_allowed=false (OR allowed but missing
 *      denominator, defensive failsafe) -> 'Insufficient evidence'
 *      label with reason hover. Legacy NOT shown.
 *   4. claim absent (healthy claims response, no row of this type) ->
 *      legacy rate renders unchanged.
 *
 * The unit label ('DMs', 'reviews') only affects the suppression
 * hover text; it does not change the gate logic.
 */

import type { VendorClaim, VendorClaimsResponse } from '../api/client'

export interface RateCardValueProps {
  claimsResponse: VendorClaimsResponse | null | undefined
  claim: VendorClaim | undefined
  legacyRate: number | null
  testIdPrefix: string
  unitLabel: string
}

export default function RateCardValue({
  claimsResponse,
  claim,
  legacyRate,
  testIdPrefix,
  unitLabel,
}: RateCardValueProps) {
  if (claimsResponse === null) {
    return (
      <span
        className="text-slate-500 italic"
        title="Claim validation API unavailable; rate not shown"
        data-testid={`${testIdPrefix}-validation-unavailable`}
      >
        Validation unavailable
      </span>
    )
  }

  if (claim) {
    if (
      claim.render_allowed &&
      claim.denominator !== null &&
      claim.denominator > 0
    ) {
      const pct = Math.round((claim.supporting_count / claim.denominator) * 100)
      return <>{`${pct}%`}</>
    }
    const reason = claim.suppression_reason
      ? `Suppressed: ${claim.suppression_reason}` +
        (claim.denominator !== null
          ? ` (${claim.supporting_count} of ${claim.denominator} ${unitLabel})`
          : '')
      : 'Suppressed: missing denominator'
    return (
      <span
        className="text-slate-500 italic"
        title={reason}
        data-testid={`${testIdPrefix}-suppressed`}
      >
        Insufficient evidence
      </span>
    )
  }

  return <>{legacyRate !== null ? `${(legacyRate * 100).toFixed(0)}%` : '--'}</>
}
