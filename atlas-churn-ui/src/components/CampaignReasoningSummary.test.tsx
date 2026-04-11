import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import CampaignReasoningSummary from './CampaignReasoningSummary'

describe('CampaignReasoningSummary', () => {
  it('returns nothing when there is no reasoning context', () => {
    const { container } = render(
      <CampaignReasoningSummary
        item={{
          company_name: 'Acme',
        }}
      />,
    )

    expect(container).toBeEmptyDOMElement()
  })

  it('renders scope, thesis, proof, account, and delta details', () => {
    render(
      <CampaignReasoningSummary
        item={{
          company_name: 'Acme',
          reasoning_scope_summary: {
            selection_strategy: 'competitive_set',
            witnesses_in_scope: 12,
            witness_mix: {
              review: 8,
              signal: 4,
            },
          },
          reasoning_atom_context: {
            top_theses: [
              {
                summary: 'Users are switching after repeated support delays.',
                why_now: 'Renewals are clustering this quarter.',
              },
            ],
            timing_windows: [
              {
                window_type: 'renewal',
                anchor: 'Q2 renewal',
                urgency: 'high',
                recommended_action: 'Reach out before procurement review.',
              },
            ],
            proof_points: [
              {
                label: 'Escalations',
                value: 5,
                interpretation: 'Support churn mentions are rising.',
              },
            ],
            account_signals: [
              {
                company: 'Northwind',
                trust_tier: 'tier_1',
                role_type: 'admin',
                buying_stage: 'evaluation',
                primary_pain: 'support',
                competitor_context: 'Zendesk',
                contract_end: '2026-06',
                decision_timeline: '30 days',
                urgency: 4,
                quote: 'We cannot keep waiting on support.',
              },
            ],
            coverage_limits: ['Limited EMEA evidence'],
          },
          reasoning_delta_summary: {
            wedge_changed: true,
            confidence_changed: true,
            top_destination_changed: true,
            theses_added: ['Support churn is now the leading wedge.'],
          },
        } as any}
      />,
    )

    expect(screen.getByText('competitive_set')).toBeInTheDocument()
    expect(screen.getByText('12 witnesses')).toBeInTheDocument()
    expect(screen.getByText('review: 8')).toBeInTheDocument()
    expect(screen.getByText('Limited EMEA evidence')).toBeInTheDocument()
    expect(screen.getByText('Top Thesis')).toBeInTheDocument()
    expect(screen.getByText('Users are switching after repeated support delays.')).toBeInTheDocument()
    expect(screen.getByText('Renewals are clustering this quarter.')).toBeInTheDocument()
    expect(screen.getByText('Timing Windows')).toBeInTheDocument()
    expect(screen.getByText('Q2 renewal')).toBeInTheDocument()
    expect(screen.getByText('renewal | high')).toBeInTheDocument()
    expect(screen.getByText('Reach out before procurement review.')).toBeInTheDocument()
    expect(screen.getByText('Proof Points')).toBeInTheDocument()
    expect(screen.getByText('Escalations')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
    expect(screen.getByText('Support churn mentions are rising.')).toBeInTheDocument()
    expect(screen.getByText('Account Signals')).toBeInTheDocument()
    expect(screen.getByText('Northwind')).toBeInTheDocument()
    expect(screen.getByText('tier_1')).toBeInTheDocument()
    expect(screen.getByText('admin | evaluation')).toBeInTheDocument()
    expect(screen.getByText('support | Zendesk')).toBeInTheDocument()
    expect(screen.getByText('2026-06 | 30 days')).toBeInTheDocument()
    expect(screen.getByText('Urgency: 4')).toBeInTheDocument()
    expect(screen.getByText('"We cannot keep waiting on support."')).toBeInTheDocument()
    expect(screen.getByText('Recent Change')).toBeInTheDocument()
    expect(screen.getByText('wedge changed')).toBeInTheDocument()
    expect(screen.getByText('confidence shifted')).toBeInTheDocument()
    expect(screen.getByText('destination competitor changed')).toBeInTheDocument()
    expect(screen.getByText('new thesis: Support churn is now the leading wedge.')).toBeInTheDocument()
  })
})
