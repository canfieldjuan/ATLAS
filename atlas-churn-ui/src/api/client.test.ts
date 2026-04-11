import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  buildReportLibraryViewScopeKey,
  downloadReportPdf,
  listWebhooks,
  normalizeReportLibraryViewFilters,
} from './client'

describe('api client helpers', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    window.history.replaceState({}, '', '/reports')
  })

  it('normalizes report library filters by trimming blanks', () => {
    expect(
      normalizeReportLibraryViewFilters({
        report_type: ' vendor_deep_dive ',
        vendor_filter: ' Salesforce ',
        quality_status: ' sales_ready ',
        freshness_state: ' ',
        review_state: '',
      }),
    ).toEqual({
      report_type: 'vendor_deep_dive',
      vendor_filter: 'Salesforce',
      quality_status: 'sales_ready',
    })
  })

  it('builds a stable scope key with defaults and slugified values', () => {
    expect(
      buildReportLibraryViewScopeKey({
        report_type: 'Vendor Deep Dive',
        vendor_filter: 'ACME / Northwind',
        quality_status: 'Sales Ready',
        freshness_state: 'Fresh Only',
      }),
    ).toBe(
      'library-view--type-vendor-deep-dive--vendor-acme-northwind--quality-sales-ready--freshness-fresh-only--review-all',
    )

    expect(buildReportLibraryViewScopeKey()).toBe(
      'library-view--type-all--vendor-all--quality-all--freshness-all--review-all',
    )
  })

  it('opens the tenant report PDF with the auth token attached', () => {
    localStorage.setItem('atlas_token', 'token-123')
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null)

    downloadReportPdf('report-42')

    expect(openSpy).toHaveBeenCalledWith(
      'http://localhost:3000/api/v1/b2b/tenant/reports/report-42/pdf?token=token-123',
      '_blank',
    )
  })

  it('uses the tenant webhook routes instead of the legacy dashboard prefix', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ webhooks: [], count: 0 }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await listWebhooks()

    const requestedUrl = String(fetchMock.mock.calls[0]?.[0] ?? '')
    const legacyDashboardPrefix = ['/api/v1', '/b2b', '/dashboard'].join('')
    expect(requestedUrl).toContain('/api/v1/b2b/tenant/webhooks')
    expect(requestedUrl).not.toContain(legacyDashboardPrefix)
  })
})
