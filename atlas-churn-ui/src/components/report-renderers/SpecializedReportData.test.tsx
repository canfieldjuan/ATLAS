import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SpecializedReportData } from './SpecializedReportData'

describe('SpecializedReportData', () => {
  it('renders thin-evidence battle cards without a headline using executive summary, talk track, and plays', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="battle_card"
          data={{
            vendor: 'Zendesk',
            quality_status: 'thin_evidence',
            executive_summary:
              'Zendesk customers are trapped in multi-month support failures that cascade into billing disputes and email delivery breakdowns.',
            talk_track: {
              opening: 'Buyers are actively pressure-testing Zendesk because contract lock in concerns keep resurfacing.',
              mid_call_pivot: 'This is not just a support problem. It is a trust problem.',
              closing: 'Let us run a quick audit of support reliability and billing predictability.',
            },
            recommended_plays: [
              {
                play: 'Target evaluators with a side-by-side evaluation on fit and switching friction.',
                key_message: 'Lead with faster evaluation clarity and fewer edge-case surprises.',
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('thin evidence')).toBeInTheDocument()
    expect(
      screen.getByText('This battle card is usable for directional validation, but the evidence base is still thin.'),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Zendesk customers are trapped in multi-month support failures/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Buyers are actively pressure-testing Zendesk/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Target evaluators with a side-by-side evaluation/i),
    ).toBeInTheDocument()
  })

  it('renders accounts-in-motion reports with witness actions', async () => {
    const user = userEvent.setup()
    const onOpenWitness = vi.fn()

    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="accounts_in_motion"
          vendorName="Zendesk"
          onOpenWitness={onOpenWitness}
          data={{
            total_accounts_in_motion: 3,
            account_pressure_summary: 'A single named account is showing early evaluation pressure.',
            account_pressure_disclaimer: 'Early account signal only.',
            account_actionability_tier: 'low',
            pricing_pressure: {
              price_complaint_rate: 0.25,
            },
            cross_vendor_context: {
              top_destination: 'Freshdesk',
            },
            reference_ids: {
              witness_ids: ['witness-1'],
            },
            accounts: [
              {
                company: 'Acme Corp',
                opportunity_score: 72,
                top_quote: 'Pricing pressure is accelerating.',
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Total Accounts')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getAllByText('Freshdesk')).toHaveLength(2)
    expect(screen.getByText('Early account signal only.')).toBeInTheDocument()
    expect(screen.getByText('Confidence tier: low')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '1 witnesses' }))
    await user.click(screen.getAllByRole('button', { name: '[1]' })[0])

    expect(onOpenWitness).toHaveBeenNthCalledWith(1, 'witness-1', 'Zendesk')
    expect(onOpenWitness).toHaveBeenNthCalledWith(2, 'witness-1', 'Zendesk')
  })


  it('links witness fallbacks to the evidence explorer with the active vendor filter', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="accounts_in_motion"
          vendorName="Zendesk"
          backTo="/report?vendor=Zendesk&ref=test-token&mode=view"
          asOfDate="2026-04-08"
          windowDays={45}
          data={{
            reference_ids: {
              witness_ids: ['witness-1'],
            },
            accounts: [
              {
                company: 'Acme Corp',
                opportunity_score: 72,
                top_quote: 'Pricing pressure is accelerating.',
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: '1 witnesses' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&back_to=%2Freport%3Fvendor%3DZendesk%26ref%3Dtest-token%26mode%3Dview',
    )
  })

  it('renders category overview arrays directly', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="category_overview"
          data={[
            {
              category: 'Helpdesk',
              dominant_pain: 'pricing',
              highest_churn_risk: 'Zendesk',
              emerging_challenger: 'Freshdesk',
              market_regime: 'replacement_heavy',
              market_shift_signal: 'Mid-market teams are actively comparing alternatives.',
            },
          ]}
        />
      </MemoryRouter>,
    )

    expect(screen.getAllByText('Helpdesk')).toHaveLength(2)
    expect(screen.getAllByText('Freshdesk')).toHaveLength(3)
    expect(screen.getByText('Mid-market teams are actively comparing alternatives.')).toBeInTheDocument()
  })

  describe('ChallengerBriefDetail head-to-head gating', () => {
    afterEach(cleanup)

    function reportSafeClaim() {
      return {
        claim_id: 'claim-report-safe',
        claim_key: 'incumbent:Google Workspace',
        claim_scope: 'competitor_pair',
        claim_type: 'direct_displacement',
        claim_text: 'Google Workspace shows direct displacement pressure toward Microsoft 365',
        target_entity: 'Microsoft 365',
        secondary_target: 'Google Workspace',
        supporting_count: 8,
        direct_evidence_count: 8,
        witness_count: 5,
        contradiction_count: 0,
        confidence: 'high',
        evidence_posture: 'usable',
        render_allowed: true,
        report_allowed: true,
        suppression_reason: null,
        evidence_links: ['review-1', 'review-2'],
        contradicting_links: [],
        as_of_date: '2026-04-26',
        analysis_window_days: 90,
        schema_version: '1',
      }
    }

    function challengerBriefData(overrides: Record<string, unknown> = {}) {
      const baseHeadToHead: Record<string, unknown> = {
        opponent: 'Microsoft 365',
        winner: 'Microsoft 365',
        loser: 'Google Workspace',
        durability: 'durable',
        confidence: 0.82,
        synthesized: false,
        conclusion: 'Microsoft 365 is winning enterprise evaluations vs Google Workspace.',
        key_insights: [
          { insight: 'Named accounts switching for collaboration depth.', evidence: 'g2:reviews' },
        ],
      }
      const headToHeadOverride = overrides.head_to_head as Record<string, unknown> | undefined
      return {
        incumbent: 'Google Workspace',
        challenger: 'Microsoft 365',
        total_target_accounts: 12,
        accounts_considering_challenger: 4,
        displacement_summary: {
          total_mentions: 22,
          signal_strength: 'moderate',
          confidence_score: 0.6,
          primary_driver: 'collaboration depth',
          source_distribution: { g2: 12, capterra: 10 },
        },
        incumbent_profile: { archetype: 'feature_gap', risk_level: 'medium' },
        challenger_advantage: {
          profile_summary: 'Stronger collaboration suite.',
          strengths: [],
          weakness_coverage: [],
          commonly_switched_from: [],
        },
        target_accounts: [],
        sales_playbook: {
          discovery_questions: [],
          landmine_questions: [],
          objection_handlers: [],
          recommended_plays: [],
        },
        integration_comparison: { shared: [], challenger_exclusive: [], incumbent_exclusive: [] },
        data_sources: { g2: true, capterra: true },
        head_to_head: { ...baseHeadToHead, ...(headToHeadOverride ?? {}) },
        ...Object.fromEntries(Object.entries(overrides).filter(([k]) => k !== 'head_to_head')),
      }
    }

    it('renders winner / durability / confidence when the head-to-head ProductClaim is report-safe', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="challenger_brief"
            data={challengerBriefData({
              head_to_head: {
                product_claim: reportSafeClaim(),
                readiness_state: 'report_safe',
                claim_validation_unavailable: false,
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Winner')).toBeInTheDocument()
      expect(screen.getAllByText('Microsoft 365').length).toBeGreaterThan(0)
      expect(screen.getByText('Durability')).toBeInTheDocument()
      // Two "Confidence" labels: displacement section + head-to-head MetricRow.
      expect(screen.getAllByText('Confidence').length).toBeGreaterThanOrEqual(2)
      expect(screen.getByText('Report-safe')).toBeInTheDocument()
      expect(screen.queryByTestId('head-to-head-strength-gate')).not.toBeInTheDocument()
      expect(
        screen.getByText('Microsoft 365 is winning enterprise evaluations vs Google Workspace.'),
      ).toBeInTheDocument()
    })

    it('hides winner / durability / confidence with a Monitor-only badge when render-safe but not report-safe', () => {
      const claim = {
        ...reportSafeClaim(),
        claim_id: 'claim-monitor',
        supporting_count: 1,
        direct_evidence_count: 1,
        witness_count: 1,
        confidence: 'low',
        render_allowed: true,
        report_allowed: false,
        suppression_reason: 'low_confidence',
      }
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="challenger_brief"
            data={challengerBriefData({
              head_to_head: {
                winner: '',
                loser: '',
                durability: '',
                confidence: null,
                synthesized: false,
                conclusion:
                  'Monitor only: direct displacement evidence for Microsoft 365 versus Google Workspace is visible in the UI but is not report-safe (low_confidence).',
                product_claim: claim,
                readiness_state: 'monitor_only',
                claim_validation_unavailable: false,
                suppression_reason: 'low_confidence',
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Winner')).not.toBeInTheDocument()
      expect(screen.queryByText('Durability')).not.toBeInTheDocument()
      // Badge text and gate fallback default both render "Monitor only".
      expect(screen.getAllByText('Monitor only').length).toBe(2)
      expect(screen.getByTestId('head-to-head-strength-gate')).toBeInTheDocument()
      expect(
        screen.getByText(/Monitor only: direct displacement evidence/i),
      ).toBeInTheDocument()
    })

    it('hides winner / durability / confidence with an Insufficient badge when the claim is suppressed', () => {
      const claim = {
        ...reportSafeClaim(),
        claim_id: 'claim-suppressed',
        supporting_count: 0,
        direct_evidence_count: 0,
        witness_count: 0,
        confidence: 'low',
        evidence_posture: 'insufficient',
        render_allowed: false,
        report_allowed: false,
        suppression_reason: 'insufficient_supporting_count',
      }
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="challenger_brief"
            data={challengerBriefData({
              head_to_head: {
                winner: '',
                loser: '',
                durability: '',
                confidence: null,
                synthesized: false,
                conclusion:
                  'Winner call suppressed: direct displacement evidence for Microsoft 365 versus Google Workspace is not render-safe (insufficient_supporting_count).',
                product_claim: claim,
                readiness_state: 'suppressed',
                claim_validation_unavailable: false,
                suppression_reason: 'insufficient_supporting_count',
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Winner')).not.toBeInTheDocument()
      expect(screen.queryByText('Durability')).not.toBeInTheDocument()
      expect(screen.getAllByText('Insufficient').length).toBe(2)
      expect(screen.getByTestId('head-to-head-strength-gate')).toBeInTheDocument()
      expect(
        screen.getByText(/Winner call suppressed/i),
      ).toBeInTheDocument()
    })

    it('hides winner / durability / confidence when ProductClaim validation is unavailable', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="challenger_brief"
            data={challengerBriefData({
              head_to_head: {
                winner: '',
                loser: '',
                durability: '',
                confidence: null,
                synthesized: false,
                conclusion:
                  'Winner call suppressed: displacement claim validation is unavailable for Microsoft 365 versus Google Workspace.',
                claim_validation_unavailable: true,
                readiness_state: 'validation_unavailable',
                suppression_reason: 'validation_unavailable',
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Winner')).not.toBeInTheDocument()
      expect(screen.queryByText('Durability')).not.toBeInTheDocument()
      expect(screen.getAllByText('Validation unavailable').length).toBe(2)
      expect(screen.getByTestId('head-to-head-strength-gate')).toBeInTheDocument()
      expect(
        screen.getByText(/displacement claim validation is unavailable/i),
      ).toBeInTheDocument()
    })

    it('falls back to a Legacy badge and hides the strength block, conclusion, and key insights when the report has no ProductClaim payload', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="challenger_brief"
            data={challengerBriefData()}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Winner')).not.toBeInTheDocument()
      expect(screen.queryByText('Durability')).not.toBeInTheDocument()
      expect(screen.getByText('Legacy')).toBeInTheDocument()
      expect(screen.getByText('Legacy fallback')).toBeInTheDocument()
      // High audit fix: legacy reports must not leak ungated winner-style narrative.
      expect(
        screen.queryByText('Microsoft 365 is winning enterprise evaluations vs Google Workspace.'),
      ).not.toBeInTheDocument()
      expect(
        screen.queryByText(/Named accounts switching for collaboration depth/i),
      ).not.toBeInTheDocument()
    })

    it('renders the Head-to-Head section with badge only when only ProductClaim metadata is present (no winner / no conclusion)', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="challenger_brief"
            data={challengerBriefData({
              head_to_head: {
                winner: '',
                loser: '',
                durability: '',
                confidence: null,
                synthesized: false,
                conclusion: '',
                key_insights: [],
                product_claim: {
                  ...reportSafeClaim(),
                  claim_id: 'claim-bare-suppressed',
                  supporting_count: 0,
                  direct_evidence_count: 0,
                  witness_count: 0,
                  evidence_posture: 'insufficient',
                  render_allowed: false,
                  report_allowed: false,
                  suppression_reason: 'insufficient_supporting_count',
                  confidence: 'low',
                },
                readiness_state: 'suppressed',
                claim_validation_unavailable: false,
                suppression_reason: 'insufficient_supporting_count',
              },
            })}
          />
        </MemoryRouter>,
      )

      // Section still renders because gate context is present.
      expect(screen.getAllByText('Insufficient').length).toBe(2)
      expect(screen.getByTestId('head-to-head-strength-gate')).toBeInTheDocument()
      expect(screen.queryByText('Winner')).not.toBeInTheDocument()
    })
  })
})
