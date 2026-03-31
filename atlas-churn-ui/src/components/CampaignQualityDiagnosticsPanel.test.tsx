import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import CampaignQualityDiagnosticsPanel from './CampaignQualityDiagnosticsPanel'

describe('CampaignQualityDiagnosticsPanel', () => {
  it('renders grouped diagnostics sections and summary cards', () => {
    render(
      <CampaignQualityDiagnosticsPanel
        data={{
          days: 14,
          top_n: 10,
          by_boundary: [{ boundary: 'manual_approval', count: 4 }],
          by_cause_type: [
            { cause_type: 'content_ignored_available_evidence', count: 3 },
          ],
          top_primary_blockers: [
            { reason: 'missing_exact_proof_term', count: 3 },
          ],
          top_missing_inputs: [
            { input: 'reasoning_reference_ids', count: 2 },
          ],
          by_target_mode: [{ target_mode: 'vendor_retention', count: 4 }],
          top_vendors: [{ vendor_name: 'Slack', count: 4 }],
        }}
      />,
    )

    expect(screen.getByText('Campaign Failure Diagnostics')).toBeInTheDocument()
    expect(screen.getByText('Top Cause')).toBeInTheDocument()
    expect(screen.getAllByText('content_ignored_available_evidence')).toHaveLength(2)
    expect(screen.getByText('Top Primary Blockers')).toBeInTheDocument()
    expect(screen.getAllByText('missing_exact_proof_term')).toHaveLength(2)
    expect(screen.getByText('Top Missing Inputs')).toBeInTheDocument()
    expect(screen.getAllByText('reasoning_reference_ids')).toHaveLength(2)
    expect(screen.getByText('Top Vendors')).toBeInTheDocument()
    expect(screen.getAllByText('Slack')).toHaveLength(2)
  })

  it('renders an empty state when diagnostics are missing', () => {
    render(<CampaignQualityDiagnosticsPanel data={null} />)

    expect(screen.getByText('No failure diagnostics available yet.')).toBeInTheDocument()
  })
})
