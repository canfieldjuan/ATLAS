import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  approveCompanySignalCandidateGroup,
  approveCompanySignalCandidateGroups,
  buildReportLibraryViewScopeKey,
  downloadReportPdf,
  fetchCompanySignalCandidateGroupSummary,
  fetchCompanySignalCandidateGroups,
  fetchCompanySignalReviewImpactSummary,
  fetchWebhookDeliverySummary,
  listWebhooks,
  listWebhookCrmPushLog,
  listWebhookDeliveries,
  normalizeReportLibraryViewFilters,
  suppressCompanySignalCandidateGroup,
  suppressCompanySignalCandidateGroups,
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

  it('uses tenant webhook activity routes with explicit filters', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ window_days: 30, active_subscriptions: 1, total_deliveries: 3, successful: 2, failed: 1, success_rate: 0.667, avg_success_duration_ms: 120, p95_success_duration_ms: 180, last_delivery_at: '2026-04-11T00:00:00Z' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ deliveries: [], count: 0 }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ pushes: [], count: 0 }),
      })
    vi.stubGlobal('fetch', fetchMock)

    await fetchWebhookDeliverySummary(30)
    await listWebhookDeliveries('wh-1', {
      success: false,
      event_type: 'signal_update',
      limit: 10,
    })
    await listWebhookCrmPushLog('wh-1', 5)

    expect(String(fetchMock.mock.calls[0]?.[0] ?? '')).toContain(
      '/api/v1/b2b/tenant/webhooks/delivery-summary?days=30',
    )
    expect(String(fetchMock.mock.calls[1]?.[0] ?? '')).toContain(
      '/api/v1/b2b/tenant/webhooks/wh-1/deliveries?success=false&event_type=signal_update&limit=10',
    )
    expect(String(fetchMock.mock.calls[2]?.[0] ?? '')).toContain(
      '/api/v1/b2b/tenant/webhooks/wh-1/crm-push-log?limit=5',
    )
  })

  it('uses the grouped review dashboard routes for queue summaries and actions', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ trend_recommendation: {}, trend_recommendation_filters: {} }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ totals: {}, top_vendors: [], pending_priority_reasons: [] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ groups: [], count: 0, candidate_bucket: 'analyst_review', review_status: 'pending' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ review_status: 'approved' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ review_status: 'suppressed' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ count: 2, groups: [] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ count: 2, groups: [] }),
      })
    vi.stubGlobal('fetch', fetchMock)

    await fetchCompanySignalReviewImpactSummary({
      window_days: 30,
      top_n: 10,
      review_action: 'approved',
    })
    await fetchCompanySignalCandidateGroupSummary({
      candidate_bucket: 'analyst_review',
      review_status: 'pending',
      top_n: 6,
    })
    await fetchCompanySignalCandidateGroups({
      candidate_bucket: 'analyst_review',
      review_status: 'pending',
      vendor_name: 'Salesforce',
      review_priority_band: 'high',
      review_priority_reason: 'cross_source_corroboration',
      limit: 10,
    })
    await approveCompanySignalCandidateGroup('group-1', {
      notes: 'Promote this group',
      trigger_rebuild: false,
    })
    await suppressCompanySignalCandidateGroup('group-2')
    await approveCompanySignalCandidateGroups({
      group_ids: ['group-1', 'group-2'],
      notes: 'Bulk approve',
    })
    await suppressCompanySignalCandidateGroups({
      group_ids: ['group-3', 'group-4'],
      trigger_rebuild: false,
    })

    expect(String(fetchMock.mock.calls[0]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-review-impact-summary',
    )
    expect(String(fetchMock.mock.calls[1]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-candidate-group-summary',
    )
    expect(String(fetchMock.mock.calls[2]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-candidate-groups',
    )
    expect(fetchMock.mock.calls[2]?.[0]).toContain('vendor_name=Salesforce')
    expect(fetchMock.mock.calls[2]?.[0]).toContain('review_priority_band=high')
    expect(fetchMock.mock.calls[2]?.[0]).toContain('review_priority_reason=cross_source_corroboration')
    expect(String(fetchMock.mock.calls[3]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-candidate-groups/group-1/approve',
    )
    expect(fetchMock.mock.calls[3]?.[1]?.body).toBe(
      JSON.stringify({ trigger_rebuild: false, notes: 'Promote this group' }),
    )
    expect(String(fetchMock.mock.calls[4]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-candidate-groups/group-2/suppress',
    )
    expect(fetchMock.mock.calls[4]?.[1]?.body).toBe(
      JSON.stringify({ trigger_rebuild: true }),
    )
    expect(String(fetchMock.mock.calls[5]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-candidate-groups/approve',
    )
    expect(fetchMock.mock.calls[5]?.[1]?.body).toBe(
      JSON.stringify({ group_ids: ['group-1', 'group-2'], trigger_rebuild: true, notes: 'Bulk approve' }),
    )
    expect(String(fetchMock.mock.calls[6]?.[0] ?? '')).toContain(
      '/api/v1/b2b/dashboard/company-signal-candidate-groups/suppress',
    )
    expect(fetchMock.mock.calls[6]?.[1]?.body).toBe(
      JSON.stringify({ group_ids: ['group-3', 'group-4'], trigger_rebuild: false }),
    )
  })
})
