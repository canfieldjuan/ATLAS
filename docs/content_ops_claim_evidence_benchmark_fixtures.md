# Content Ops Claim Evidence Benchmark Fixtures

Issue #1435 benchmarks `verify_claim_evidence` before any model-filled
structured-judgment slot can enter the verifier rubric. The benchmark input is a
decoded list of labeled claim/evidence triples. The future runner may load this
from JSON or JSONL, but the package contract starts after parsing.

## Row Shape

Each row is an object with these fields:

```json
{
  "triple_id": "real-001",
  "claim_id": "claim-churn-001",
  "evidence_quote": "Customers reduced manual escalation work by 31% after rollout.",
  "source_id": "case-study-acme-q4",
  "expected_supports": true,
  "difficulty": "easy"
}
```

Field rules:

- `triple_id`: unique non-empty string for this benchmark row.
- `claim_id`: non-empty string that identifies the claim under test.
- `evidence_quote`: non-empty quote or excerpt the model will judge.
- `source_id`: non-empty source identifier for traceability.
- `expected_supports`: operator label; `true` means the evidence supports the
  claim, `false` means it does not.
- `difficulty`: either `easy` or `hard`.

## Final Composition

The final #1435 benchmark set targets 40 rows:

| Bucket | Count |
|---|---:|
| Easy support | 15 |
| Easy non-support | 15 |
| Hard cases | 10 |

Seed sets do not need to hit these counts. The validator supports seed mode for
early 5 to 10 real registry triples and final mode for the completed benchmark.

## Hard Cases

Hard rows should feel like realistic B2B SaaS marketing copy and include cases
where keyword overlap is not enough:

- paraphrased support;
- partial support;
- evidence supports a narrower claim;
- evidence supports a broader claim;
- evidence shares keywords but contradicts or misses the claim logic.

## Out Of Scope

This fixture does not include model responses, prompts, provider settings,
tokens, customer draft content, or verifier/MCP wiring. The future runner owns
file loading, provider execution, and result artifacts.
