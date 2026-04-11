import { describe, expect, it } from 'vitest'
import {
  humanLabel,
  isSpecializedReportType,
  REPORT_SCALAR_KEYS,
  SPECIALIZED_REPORT_TYPES,
} from './reportConstants'

describe('reportConstants', () => {
  it('recognizes specialized report types and preserves the exported set', () => {
    expect(SPECIALIZED_REPORT_TYPES).toContain('vendor_deep_dive')
    expect(isSpecializedReportType('vendor_deep_dive')).toBe(true)
    expect(isSpecializedReportType('exploratory_overview')).toBe(false)
  })

  it('formats known and fallback field labels', () => {
    expect(REPORT_SCALAR_KEYS.has('vendor_name')).toBe(true)
    expect(humanLabel('avg_urgency')).toBe('Avg Urgency')
    expect(humanLabel('custom_signal_density')).toBe('Custom Signal Density')
  })
})
