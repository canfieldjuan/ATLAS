import { describe, expect, it } from 'vitest'
import {
  normalizeReportObject,
  normalizeUnknown,
  normalizeVendorProfile,
} from './reportNormalization'

describe('reportNormalization', () => {
  it('parses JSONish values and normalizes insight arrays', () => {
    const normalized = normalizeReportObject({
      top_alternatives: '["HubSpot","HubSpot","Salesforce"]',
      key_insights: '["Fast setup","Low switching friction"]',
      source_distribution: '{"review": 8, "signal": 3}',
    })

    expect(normalized.top_alternatives).toEqual(['HubSpot', 'Salesforce'])
    expect(normalized.key_insights).toEqual([
      { insight: 'Fast setup', evidence: '' },
      { insight: 'Low switching friction', evidence: '' },
    ])
    expect(normalized.source_distribution).toEqual({ review: 8, signal: 3 })
  })

  it('salvages comma-delimited list strings for known keys', () => {
    expect(normalizeUnknown('HubSpot, Salesforce, HubSpot', 'top_alternatives')).toEqual([
      'HubSpot',
      'Salesforce',
    ])
    expect(normalizeUnknown('Plain text sentence', 'summary')).toBe('Plain text sentence')
  })

  it('normalizes vendor profile JSON-like fields', () => {
    const profile = normalizeVendorProfile({
      churn_signal: '{"top_feature_gaps":"[\\"API\\",\\"API\\",\\"UX\\"]"}',
      high_intent_companies: '[{"company":"Acme","urgency":4}]',
      pain_distribution: '{"support": 3}',
    } as any)

    expect(profile.churn_signal).toEqual({
      top_feature_gaps: ['API', 'UX'],
    })
    expect(profile.high_intent_companies).toEqual([{ company: 'Acme', urgency: 4 }])
    expect(profile.pain_distribution).toEqual({ support: 3 })
  })
})
