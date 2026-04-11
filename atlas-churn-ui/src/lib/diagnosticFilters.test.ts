import { describe, expect, it } from 'vitest'
import {
  coerceDiagnosticParam,
  DEFAULT_DIAGNOSTIC_DAYS,
  DEFAULT_DIAGNOSTICS_TOP_N,
  DIAGNOSTIC_DAY_OPTIONS,
  DIAGNOSTIC_TOP_N_OPTIONS,
} from './diagnosticFilters'

describe('diagnosticFilters', () => {
  it('keeps allowed numeric params', () => {
    expect(coerceDiagnosticParam('30', DIAGNOSTIC_DAY_OPTIONS, DEFAULT_DIAGNOSTIC_DAYS)).toBe(30)
    expect(coerceDiagnosticParam('20', DIAGNOSTIC_TOP_N_OPTIONS, DEFAULT_DIAGNOSTICS_TOP_N)).toBe(20)
  })

  it('falls back for non-numeric or disallowed values', () => {
    expect(coerceDiagnosticParam('abc', DIAGNOSTIC_DAY_OPTIONS, DEFAULT_DIAGNOSTIC_DAYS)).toBe(
      DEFAULT_DIAGNOSTIC_DAYS,
    )
    expect(coerceDiagnosticParam('45', DIAGNOSTIC_DAY_OPTIONS, DEFAULT_DIAGNOSTIC_DAYS)).toBe(
      DEFAULT_DIAGNOSTIC_DAYS,
    )
    expect(coerceDiagnosticParam(null, DIAGNOSTIC_TOP_N_OPTIONS, DEFAULT_DIAGNOSTICS_TOP_N)).toBe(
      DEFAULT_DIAGNOSTICS_TOP_N,
    )
  })
})
