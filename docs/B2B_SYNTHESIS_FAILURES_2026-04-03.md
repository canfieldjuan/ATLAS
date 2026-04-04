# B2B Synthesis Failures - 2026-04-03

Latest vendor synthesis rebuild completed with `53/56` vendors refreshed successfully.

Remaining stale vendors:

- `Help Scout`
  - Latest successful row still from `2026-04-01 21:25:05.869267-05`
  - Failure stage: `validation`
  - Failure reason: `packet_artifacts.witness_pack: witness selection fell back to generic excerpts because the specific witness pool was too thin`
  - Tokens spent in failed run: `33771`
  - Attempts used: `2`

- `Metabase`
  - Latest successful row still from `2026-04-01 14:38:04.701732-05`
  - Failure stage: `validation`
  - Failure reason: `packet_artifacts.witness_pack: witness selection fell back to generic excerpts because the specific witness pool was too thin`
  - Tokens spent in failed run: `38086`
  - Attempts used: `2`

- `Microsoft Defender for Endpoint`
  - Latest successful row still from `2026-04-01 14:39:05.943117-05`
  - Failure stage: `validation`
  - Failure reason: `causal_narrative.data_gaps: contradictions on [api_limitations] not reflected in data_gaps`
  - Tokens spent in failed run: `32163`
  - Attempts used: `2`

Context:

- Root-cause config issue fixed before this rebuild: removed `ATLAS_B2B_CHURN_REASONING_SYNTHESIS_MODEL=anthropic/claude-haiku-4-5` from `.env.local`
- Canonical synthesis now falls back to the standard OpenRouter reasoning model (`anthropic/claude-sonnet-4-5`)
- Cross-vendor synthesis still has separate input-budget rejections and is not included in the vendor failure list above
