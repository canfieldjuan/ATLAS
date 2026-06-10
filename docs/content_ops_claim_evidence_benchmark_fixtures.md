# Content Ops Claim Evidence Benchmark Fixtures

Issue #1435 benchmarks `verify_claim_evidence` before any model-filled
structured-judgment slot can enter the verifier rubric. The benchmark input is a
list of labeled claim/evidence triples. The package loader accepts raw JSON
array text or JSONL object streams and then applies the same decoded row
contract.

## Row Shape

Each row is an object with these fields:

```json
{
  "triple_id": "real-001",
  "claim_id": "claim-churn-001",
  "claim_text": "Customers reduced manual escalation work by 31% after rollout.",
  "evidence_quote": "Customers reduced manual escalation work by 31% after rollout.",
  "source_id": "case-study-acme-q4",
  "expected_supports": true,
  "difficulty": "easy"
}
```

Field rules:

- `triple_id`: unique non-empty string for this benchmark row.
- `claim_id`: non-empty identifier for traceability, deduplication, or registry
  linkage.
- `claim_text`: non-empty claim statement the model judges against the evidence.
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

## Fixture Text Formats

JSON fixture text is one array of row objects:

```json
[
  {
    "triple_id": "real-001",
    "claim_id": "claim-churn-001",
    "claim_text": "Customers reduced manual escalation work by 31% after rollout.",
    "evidence_quote": "Customers reduced manual escalation work by 31% after rollout.",
    "source_id": "case-study-acme-q4",
    "expected_supports": true,
    "difficulty": "easy"
  }
]
```

JSONL fixture text is one row object per non-empty line:

```jsonl
{"triple_id":"real-001","claim_id":"claim-churn-001","claim_text":"Customers reduced manual escalation work by 31% after rollout.","evidence_quote":"Customers reduced manual escalation work by 31% after rollout.","source_id":"case-study-acme-q4","expected_supports":true,"difficulty":"easy"}
{"triple_id":"hard-001","claim_id":"claim-integration-002","claim_text":"The platform syncs Salesforce opportunities natively.","evidence_quote":"The platform exports CSV files for CRM import.","source_id":"docs-export","expected_supports":false,"difficulty":"hard"}
```

JSONL lines must be objects. Arrays belong in JSON fixture text, not JSONL, so
line numbers remain unambiguous when the loader reports malformed input.

## Local Validation

Validate a draft fixture file before running any model benchmark:

```bash
python scripts/validate_content_ops_claim_evidence_fixture.py path/to/fixture.json
python scripts/validate_content_ops_claim_evidence_fixture.py path/to/fixture.jsonl --format jsonl
python scripts/validate_content_ops_claim_evidence_fixture.py path/to/final.json --require-final-shape
```

The command prints a JSON envelope with `ok`, `errors`, `triple_count`,
`easy_supports_count`, `easy_not_supports_count`, and `hard_count`. It exits
zero only when the fixture is valid. It does not call models, write benchmark
results, or expose verifier/MCP behavior.

## Prompt And Response Contract

The structured-witness contract is versioned as `verify_claim_evidence.v1`.
For each benchmark row, the future provider runner will render a prompt with:

- the claim text;
- the claim id for traceability;
- the evidence quote;
- the source id;
- the difficulty bucket.

The prompt asks the witness to decide whether the evidence quote supports the
claim under test. It explicitly tells the witness not to decide whether the
claim is generally true and not to use outside knowledge.

Responses must match this strict JSON shape:

```json
{
  "supports": true,
  "confidence": 4,
  "reason": "The quote directly states the measured outcome."
}
```

Field rules:

- `supports`: boolean support judgment.
- `confidence`: integer from 1 through 5. Downstream scoring only credits high
  confidence responses.
- `reason`: non-empty, non-whitespace rationale grounded in the evidence quote.

Extra response fields are not part of the contract.

## Runner Harness

The benchmark runner harness is intentionally provider-injected. It renders the
prompt/schema contract for each valid fixture row, calls a supplied provider
boundary, and decodes the returned response through the same structured-response
validator documented above.

Runner output is a per-model in-memory result:

- valid rows expose decoded responses keyed by `triple_id` for the existing
  scorer;
- malformed provider responses are recorded as row errors and excluded from the
  response map;
- provider exceptions are recorded by exception class name and do not stop later
  rows from running.

The harness does not choose models, read credentials, call network providers,
write benchmark result files, or expose verifier/MCP behavior.

## Hard Cases

Hard rows should feel like realistic B2B SaaS marketing copy and include cases
where keyword overlap is not enough:

- paraphrased support;
- partial support;
- evidence supports a narrower claim;
- evidence supports a broader claim;
- evidence shares keywords but contradicts or misses the claim logic.

## Out Of Scope

This fixture does not include model responses, provider settings, tokens,
customer draft content, or verifier/MCP wiring. The future runner owns provider
execution and result artifacts.
