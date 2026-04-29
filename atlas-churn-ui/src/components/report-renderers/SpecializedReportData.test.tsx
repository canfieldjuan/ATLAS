import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SpecializedReportData } from './SpecializedReportData'

describe('SpecializedReportData', () => {
  afterEach(cleanup)

  function reportSafeCompetitorPairClaim(overrides: Record<string, unknown> = {}) {
    return {
      claim_id: 'claim-cross-vendor-report-safe',
      claim_key: 'incumbent:Zendesk',
      claim_scope: 'competitor_pair',
      claim_type: 'direct_displacement',
      claim_text: 'Zendesk shows direct displacement pressure toward Freshdesk',
      target_entity: 'Freshdesk',
      secondary_target: 'Zendesk',
      supporting_count: 4,
      direct_evidence_count: 4,
      witness_count: 3,
      contradiction_count: 0,
      denominator: 4,
      sample_size: 4,
      has_grounded_evidence: true,
      confidence: 'medium',
      evidence_posture: 'usable',
      render_allowed: true,
      report_allowed: true,
      suppression_reason: null,
      evidence_links: ['review-1', 'review-2'],
      contradicting_links: [],
      as_of_date: '2026-04-26',
      analysis_window_days: 90,
      schema_version: '1',
      ...overrides,
    }
  }

  it('fails closed for battle-card talk tracks and plays without ProductClaim context', () => {
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
      screen.queryByText(/Buyers are actively pressure-testing Zendesk/i),
    ).not.toBeInTheDocument()
    expect(screen.getByTestId('battle-card-talk-track-gate')).toHaveTextContent(
      'Legacy/unvalidated talk track',
    )
    expect(
      screen.queryByText(/Target evaluators with a side-by-side evaluation/i),
    ).not.toBeInTheDocument()
    expect(screen.getByTestId('recommended-play-0-gate')).toHaveTextContent(
      'Legacy/unvalidated play',
    )
  })

  it('renders battle-card talk tracks and plays when their ProductClaims are report-safe', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="battle_card"
          data={{
            vendor: 'Zendesk',
            talk_track: {
              opening: 'Buyers are actively pressure-testing Zendesk because reliability keeps resurfacing.',
              mid_call_pivot: 'This is a trust problem.',
              closing: 'Let us run a quick audit.',
              product_claim: reportSafeCompetitorPairClaim({ claim_id: 'claim-talk-track' }),
            },
            recommended_plays: [
              {
                play: 'Target evaluators with a side-by-side evaluation on fit.',
                key_message: 'Lead with faster clarity.',
                product_claim: reportSafeCompetitorPairClaim({ claim_id: 'claim-play-1' }),
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getAllByText('Report-safe')).toHaveLength(2)
    expect(screen.queryByTestId('battle-card-talk-track-gate')).not.toBeInTheDocument()
    expect(screen.queryByTestId('recommended-play-0-gate')).not.toBeInTheDocument()
    expect(
      screen.getByText(/Buyers are actively pressure-testing Zendesk/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Target evaluators with a side-by-side evaluation/i),
    ).toBeInTheDocument()
  })

  it('fails closed for battle-card cross-vendor battles without ProductClaim context', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="battle_card"
          data={{
            vendor: 'Zendesk',
            cross_vendor_battles: [
              {
                opponent: 'Freshdesk',
                winner: 'Freshdesk',
                loser: 'Zendesk',
                confidence: 0.72,
                durability: 'durable',
                conclusion: 'Freshdesk is winning enterprise evaluations from Zendesk.',
                key_insights: [{ insight: 'Buyers are switching for support quality.' }],
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Cross-Vendor Battles')).toBeInTheDocument()
    expect(screen.getByText('Legacy')).toBeInTheDocument()
    expect(screen.getByTestId('cross-vendor-battle-0-gate')).toHaveTextContent(
      'Legacy/unvalidated battle',
    )
    expect(screen.queryByText('winner: Freshdesk')).not.toBeInTheDocument()
    expect(
      screen.queryByText('Freshdesk is winning enterprise evaluations from Zendesk.'),
    ).not.toBeInTheDocument()
  })

  it('renders battle-card cross-vendor battles when the ProductClaim is report-safe', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="battle_card"
          data={{
            vendor: 'Zendesk',
            cross_vendor_battles: [
              {
                opponent: 'Freshdesk',
                winner: 'Freshdesk',
                loser: 'Zendesk',
                confidence: 0.72,
                durability: 'durable',
                conclusion: 'Freshdesk is winning enterprise evaluations from Zendesk.',
                key_insights: [{ insight: 'Buyers are switching for support quality.' }],
                product_claim: reportSafeCompetitorPairClaim(),
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Report-safe')).toBeInTheDocument()
    expect(screen.queryByTestId('cross-vendor-battle-0-gate')).not.toBeInTheDocument()
    expect(screen.getByText('winner: Freshdesk')).toBeInTheDocument()
    expect(
      screen.getByText('Freshdesk is winning enterprise evaluations from Zendesk.'),
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
    expect(screen.getByText('Freshdesk')).toBeInTheDocument()
    expect(screen.getByText('Mid-market teams are actively comparing alternatives.')).toBeInTheDocument()
  })

  it('fails closed for weekly churn feed displacement targets without ProductClaim context', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="weekly_churn_feed"
          data={[
            {
              vendor: 'Zendesk',
              pain_breakdown: [],
              top_displacement_targets: [
                {
                  competitor: 'Intercom',
                  mentions: 12,
                },
              ],
            },
          ]}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Displacement Targets')).toBeInTheDocument()
    expect(screen.getByText('Legacy')).toBeInTheDocument()
    expect(screen.getByTestId('weekly-displacement-target-0-0-gate')).toHaveTextContent(
      'Legacy/unvalidated target',
    )
    expect(screen.queryByText('Intercom')).not.toBeInTheDocument()
    expect(screen.queryByText('12')).not.toBeInTheDocument()
  })

  it('renders weekly churn feed displacement targets when the ProductClaim is report-safe', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="weekly_churn_feed"
          data={[
            {
              vendor: 'Zendesk',
              pain_breakdown: [],
              top_displacement_targets: [
                {
                  competitor: 'Intercom',
                  mentions: 12,
                  product_claim: reportSafeCompetitorPairClaim({ claim_id: 'claim-weekly-target' }),
                },
              ],
            },
          ]}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Displacement Targets')).toBeInTheDocument()
    expect(screen.getByText('Report-safe')).toBeInTheDocument()
    expect(screen.queryByTestId('weekly-displacement-target-0-0-gate')).not.toBeInTheDocument()
    expect(screen.getByText('Intercom')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
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

  describe('BattleCardDetail displacement reasoning section', () => {
    afterEach(cleanup)

    function reportSafeBattleCardClaim() {
      return {
        claim_id: 'claim-bc-report-safe',
        claim_key: 'incumbent:Zendesk',
        claim_scope: 'competitor_pair',
        claim_type: 'direct_displacement',
        claim_text: 'Zendesk shows direct displacement pressure toward Freshdesk',
        target_entity: 'Freshdesk',
        secondary_target: 'Zendesk',
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

    function battleCardData(displacementReasoning: Record<string, unknown> | null) {
      const card: Record<string, unknown> = {
        vendor: 'Zendesk',
        category: 'Helpdesk',
        churn_pressure_score: 72,
        executive_summary: 'Zendesk customers under pressure.',
      }
      if (displacementReasoning !== null) {
        card.reasoning_contracts = { displacement_reasoning: displacementReasoning }
      }
      return card
    }

    it('renders migration_proof and customer_winning_pattern when both fields are report-safe', () => {
      const claim = reportSafeBattleCardClaim()
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: true,
                suppression_reason: null,
                product_claims: [claim],
              },
              migration_proof: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: true,
                suppression_reason: null,
                confidence: 'high',
                switching_is_real: true,
                top_destination: 'Freshdesk',
                switch_volume: { value: 4 },
                product_claims: [claim],
              },
              customer_winning_pattern: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: true,
                suppression_reason: null,
                confidence: 'high',
                summary: 'Freshdesk is winning named evaluations against Zendesk.',
                product_claims: [claim],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.getByText('Migration Proof')).toBeInTheDocument()
      expect(screen.getByText('Customer Winning Pattern')).toBeInTheDocument()
      expect(screen.getByText('Switching Real')).toBeInTheDocument()
      expect(screen.getByText('Top Destination')).toBeInTheDocument()
      expect(screen.getByText('Freshdesk')).toBeInTheDocument()
      expect(screen.getByText('Switch Volume')).toBeInTheDocument()
      expect(
        screen.getByText('Freshdesk is winning named evaluations against Zendesk.'),
      ).toBeInTheDocument()
      expect(screen.getByText('Report-safe')).toBeInTheDocument()
      expect(screen.queryByTestId('battle-card-migration-proof-gate')).not.toBeInTheDocument()
      expect(
        screen.queryByTestId('battle-card-customer-winning-pattern-gate'),
      ).not.toBeInTheDocument()
    })

    it('strips migration_proof and customer_winning_pattern content when monitor-only, surfacing gate_message', () => {
      const claim = {
        ...reportSafeBattleCardClaim(),
        claim_id: 'claim-bc-monitor',
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
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'monitor_only',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                product_claims: [claim],
              },
              migration_proof: {
                readiness_state: 'monitor_only',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                gate_message:
                  'Battle-card displacement proof is monitor-only: direct displacement evidence is visible in the UI but not report-safe (low_confidence).',
                product_claims: [claim],
              },
              customer_winning_pattern: {
                readiness_state: 'monitor_only',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                gate_message:
                  'Battle-card displacement proof is monitor-only: direct displacement evidence is visible in the UI but not report-safe (low_confidence).',
                product_claims: [claim],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.queryByText('Switching Real')).not.toBeInTheDocument()
      expect(screen.queryByText('Top Destination')).not.toBeInTheDocument()
      expect(screen.queryByText('Switch Volume')).not.toBeInTheDocument()
      expect(screen.getByText('Monitor only')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-migration-proof-gate')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-customer-winning-pattern-gate')).toBeInTheDocument()
      expect(
        screen.getAllByText(/monitor-only: direct displacement evidence/i).length,
      ).toBeGreaterThanOrEqual(1)
    })

    it('strips content with an Insufficient badge when the displacement gate is suppressed', () => {
      const claim = {
        ...reportSafeBattleCardClaim(),
        claim_id: 'claim-bc-suppressed',
        supporting_count: 0,
        direct_evidence_count: 0,
        witness_count: 0,
        evidence_posture: 'insufficient',
        render_allowed: false,
        report_allowed: false,
        suppression_reason: 'insufficient_supporting_count',
        confidence: 'low',
      }
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'suppressed',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'insufficient_supporting_count',
                product_claims: [claim],
              },
              migration_proof: {
                readiness_state: 'suppressed',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'insufficient_supporting_count',
                gate_message:
                  'Battle-card displacement proof suppressed: direct displacement evidence is not report-safe (insufficient_supporting_count).',
                product_claims: [claim],
              },
              customer_winning_pattern: {
                readiness_state: 'suppressed',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'insufficient_supporting_count',
                gate_message:
                  'Battle-card displacement proof suppressed: direct displacement evidence is not report-safe (insufficient_supporting_count).',
                product_claims: [claim],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.queryByText('Switching Real')).not.toBeInTheDocument()
      expect(screen.queryByText('Top Destination')).not.toBeInTheDocument()
      expect(screen.getByText('Insufficient')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-migration-proof-gate')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-customer-winning-pattern-gate')).toBeInTheDocument()
      expect(
        screen.getAllByText(/displacement proof suppressed/i).length,
      ).toBeGreaterThanOrEqual(1)
    })

    it('strips content with a Validation-unavailable badge when ProductClaim validation is unavailable', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'validation_unavailable',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'validation_unavailable',
                product_claims: [],
              },
              migration_proof: {
                readiness_state: 'validation_unavailable',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'validation_unavailable',
                gate_message:
                  'Battle-card displacement proof suppressed: ProductClaim validation is unavailable.',
                product_claims: [],
              },
              customer_winning_pattern: {
                readiness_state: 'validation_unavailable',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'validation_unavailable',
                gate_message:
                  'Battle-card displacement proof suppressed: ProductClaim validation is unavailable.',
                product_claims: [],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.queryByText('Switching Real')).not.toBeInTheDocument()
      expect(screen.queryByText('Top Destination')).not.toBeInTheDocument()
      expect(screen.getByText('Validation unavailable')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-migration-proof-gate')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-customer-winning-pattern-gate')).toBeInTheDocument()
      expect(
        screen.getAllByText(/ProductClaim validation is unavailable/i).length,
      ).toBeGreaterThanOrEqual(1)
    })

    it('omits the entire Displacement Reasoning section when no product_claim_gate exists (legacy / fail-closed)', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData(null)}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Displacement Reasoning')).not.toBeInTheDocument()
      expect(screen.queryByText('Migration Proof')).not.toBeInTheDocument()
      expect(screen.queryByText('Customer Winning Pattern')).not.toBeInTheDocument()
      expect(
        screen.queryByTestId('battle-card-migration-proof-gate'),
      ).not.toBeInTheDocument()
      expect(
        screen.queryByTestId('battle-card-customer-winning-pattern-gate'),
      ).not.toBeInTheDocument()
    })

    it('omits the section when reasoning_contracts.displacement_reasoning has fields but no product_claim_gate (defensive fail-closed)', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              migration_proof: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: true,
                suppression_reason: null,
                confidence: 'high',
                switching_is_real: true,
                top_destination: 'Freshdesk',
                product_claims: [],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Displacement Reasoning')).not.toBeInTheDocument()
      expect(screen.queryByText('Migration Proof')).not.toBeInTheDocument()
    })

    it('renders the Insufficient badge when section gate is suppressed with empty product_claims', () => {
      // Backend produces this when claim_rows=[]: aggregator returns
      // readiness_state="suppressed" with product_claims=[]. The section badge
      // must derive from readiness_state, not from a representative claim.
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'suppressed',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'insufficient_supporting_count',
                product_claims: [],
              },
              migration_proof: {
                readiness_state: 'suppressed',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'insufficient_supporting_count',
                gate_message: 'Battle-card displacement proof suppressed: no validated rows.',
                product_claims: [],
              },
              customer_winning_pattern: {
                readiness_state: 'suppressed',
                render_allowed: false,
                report_allowed: false,
                suppression_reason: 'insufficient_supporting_count',
                gate_message: 'Battle-card displacement proof suppressed: no validated rows.',
                product_claims: [],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.getByText('Insufficient')).toBeInTheDocument()
      expect(screen.queryByText('Legacy')).not.toBeInTheDocument()
      expect(screen.getByTestId('battle-card-migration-proof-gate')).toBeInTheDocument()
    })

    it('reads displacement_reasoning from the top-level card key when reasoning_contracts is absent', () => {
      const claim = reportSafeBattleCardClaim()
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={{
              vendor: 'Zendesk',
              category: 'Helpdesk',
              displacement_reasoning: {
                product_claim_gate: {
                  readiness_state: 'report_safe',
                  render_allowed: true,
                  report_allowed: true,
                  suppression_reason: null,
                  product_claims: [claim],
                },
                migration_proof: {
                  readiness_state: 'report_safe',
                  render_allowed: true,
                  report_allowed: true,
                  suppression_reason: null,
                  confidence: 'high',
                  top_destination: 'Freshdesk',
                  product_claims: [claim],
                },
              },
            }}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.getByText('Migration Proof')).toBeInTheDocument()
      expect(screen.getByText('Top Destination')).toBeInTheDocument()
      expect(screen.getByText('Report-safe')).toBeInTheDocument()
    })

    it('honors per-field readiness independently (one report-safe, one monitor-only)', () => {
      const reportSafe = reportSafeBattleCardClaim()
      const monitorClaim = {
        ...reportSafe,
        claim_id: 'claim-bc-monitor-mixed',
        render_allowed: true,
        report_allowed: false,
        suppression_reason: 'low_confidence',
        confidence: 'low',
      }
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: true,
                suppression_reason: null,
                product_claims: [reportSafe],
              },
              migration_proof: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: true,
                suppression_reason: null,
                confidence: 'high',
                switching_is_real: true,
                top_destination: 'Freshdesk',
                product_claims: [reportSafe],
              },
              customer_winning_pattern: {
                readiness_state: 'monitor_only',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                gate_message:
                  'Battle-card customer winning pattern is monitor-only (low_confidence).',
                product_claims: [monitorClaim],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Migration Proof')).toBeInTheDocument()
      expect(screen.getByText('Switching Real')).toBeInTheDocument()
      expect(screen.getByText('Top Destination')).toBeInTheDocument()
      expect(screen.queryByTestId('battle-card-migration-proof-gate')).not.toBeInTheDocument()
      expect(screen.getByTestId('battle-card-customer-winning-pattern-gate')).toBeInTheDocument()
      expect(
        screen.getByText(/customer winning pattern is monitor-only/i),
      ).toBeInTheDocument()
    })

    it('falls through to the gate fallback when readiness_state says report_safe but report_allowed is false (malformed payload)', () => {
      // Defense in depth: trust the boolean gate, not just the readiness label.
      const claim = {
        ...reportSafeBattleCardClaim(),
        claim_id: 'claim-bc-malformed',
        render_allowed: true,
        report_allowed: false,
        suppression_reason: 'low_confidence',
      }
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'monitor_only',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                product_claims: [claim],
              },
              migration_proof: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                gate_message: 'Migration proof gated despite report_safe label.',
                confidence: 'high',
                switching_is_real: true,
                top_destination: 'Freshdesk',
                product_claims: [claim],
              },
              customer_winning_pattern: {
                readiness_state: 'report_safe',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                gate_message: 'Customer winning pattern gated despite report_safe label.',
                confidence: 'high',
                summary: 'Should not render.',
                product_claims: [claim],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.queryByText('Switching Real')).not.toBeInTheDocument()
      expect(screen.queryByText('Top Destination')).not.toBeInTheDocument()
      expect(screen.queryByText('Should not render.')).not.toBeInTheDocument()
      expect(screen.getByTestId('battle-card-migration-proof-gate')).toBeInTheDocument()
      expect(screen.getByTestId('battle-card-customer-winning-pattern-gate')).toBeInTheDocument()
      expect(
        screen.getByText('Migration proof gated despite report_safe label.'),
      ).toBeInTheDocument()
      expect(
        screen.getByText('Customer winning pattern gated despite report_safe label.'),
      ).toBeInTheDocument()
    })

    it('renders header + badge with empty grid when product_claim_gate is present but both fields are absent', () => {
      render(
        <MemoryRouter>
          <SpecializedReportData
            reportType="battle_card"
            data={battleCardData({
              product_claim_gate: {
                readiness_state: 'monitor_only',
                render_allowed: true,
                report_allowed: false,
                suppression_reason: 'low_confidence',
                product_claims: [],
              },
            })}
          />
        </MemoryRouter>,
      )

      expect(screen.getByText('Displacement Reasoning')).toBeInTheDocument()
      expect(screen.getByText('Monitor only')).toBeInTheDocument()
      expect(screen.queryByText('Migration Proof')).not.toBeInTheDocument()
      expect(screen.queryByText('Customer Winning Pattern')).not.toBeInTheDocument()
      expect(screen.queryByTestId('battle-card-migration-proof-gate')).not.toBeInTheDocument()
      expect(
        screen.queryByTestId('battle-card-customer-winning-pattern-gate'),
      ).not.toBeInTheDocument()
    })
  })
})
