# PR-Deflection-Hybrid-Question-Clustering

## Why this slice exists

#1504 measured that the question sub-clustering in
`extracted_content_pipeline/ticket_faq_markdown.py` (`_question_subclusters`)
ships a MinHash-LSH pre-filter (`_MINHASH_BAND_ROWS` = 4) that delivers an effective
~0.71 Jaccard bar, not the documented `_SUBCLUSTER_JACCARD_THRESHOLD` of 1/3. Two
consequences: the delivered rule silently differs from the documented one (counts are
a conservative floor, but order-sensitive), and -- the structural ceiling -- every
lexical option misses reworded repeats that share no tokens ("how do I get my money
back?" vs "what is the refund process?"). That long tail is exactly the inventory a
deflection report sells.

This is the hybrid clustering agreed in discussion #1507 / #1504 Option 5: an exact,
auditable lexical floor that makes the documented 1/3 bar literally true and
order-insensitive, plus an embedding recall booster -- injected through a host port,
so the standalone package stays model-free -- that recovers the no-shared-token
rewordings the floor cannot reach. Each embedding-driven merge carries its cosine
score so the merge stays inspectable.

**Status: RFC.** This PR adds the plan only, for approach review before any code. The
implementation lands in a follow-up PR (file set + diff size below in Deferred), so
this PR's audited Files-touched and Diff-size reflect just the plan doc.

## Scope (this PR)

This PR adds the plan doc only (RFC, for approach review). The numbered items below
are the proposed implementation slice that lands in the follow-up PR.

Ownership lane: deflection/clustering
Slice phase: Production hardening

1. `extracted_content_pipeline/ticket_faq_markdown.py`: replace the MinHash-LSH
   pre-filter in `_question_subclusters` with a prefix-filtered all-pairs join
   (PPJoin-style: exact-gist dedupe, rare-token-first prefix, length-ratio filter,
   exact `_jaccard` verification against `_SUBCLUSTER_JACCARD_THRESHOLD`, same
   union-find single-link semantics). Remove `_MINHASH_PERMUTATIONS` /
   `_MINHASH_BAND_ROWS` / `_minhash_signature` once unreferenced.
2. A new `embedding_port` module (under the `extracted_content_pipeline/` package):
   an `EmbeddingPort` Protocol plus a pure `cosine` helper. Thread an optional
   `embedding_port` through `_question_subclusters` and its caller; when it is `None`,
   behavior is exactly the lexical floor.
3. Embedding recall booster: when a port is injected, embed the per-group question
   gists, nominate the not-yet-merged pairs whose exact cosine clears a calibrated
   threshold, merge them, and record the cosine on the merge (vs the lexical merges'
   shared-token evidence) in the sub-cluster diagnostics.
4. Host wiring: a `HostEmbeddingPort` adapter in `atlas_brain/_content_ops_infrastructure.py`
   wrapping `atlas_brain/services/embedding/sentence_transformer.py` (the
   `get_embedding_service` batch path, run in a worker thread for async parity,
   mirroring `HostLLMClient`), injected via `atlas_brain/_content_ops_services.py`.

### Review Contract
- Acceptance criteria (for the implementation PR): lexical floor reproduces the
  documented 1/3 bar (>= 1/3 merges, < 1/3 never does) and is order-insensitive (same
  rows in any order -> identical clusters); with no port injected, output is
  byte-identical to the floor; with a deterministic stub port, a reworded
  no-shared-token pair merges and carries a cosine value; same input + pinned model +
  threshold -> identical clusters.
- Model-source acceptance criterion (R11): the embedding path pins the model name +
  revision and an offline / cache contract (e.g. local_files_only) so the determinism
  and "no network beyond the local CPU model" claims actually hold. The shipped
  `atlas_brain/services/embedding/sentence_transformer.py` defaults to all-MiniLM-L6-v2
  with no revision/offline pin, so the implementation must add this at the adapter or
  service level -- or the plan must instead state that first-run model download is
  allowed and exactly how it is gated (env/config) with inference offline thereafter.
- Host-port acceptance criterion (R2): the host adapter + services-bundle injection are
  covered by their own host-side wiring tests (the port returns embeddings off the
  worker thread; the bundle injects it; offline/pinned behavior is asserted), not only
  the extracted-package clustering tests -- the port boundary is the risky seam.
- Approach decisions this RFC asks the reviewer to weigh: embeddings-via-hybrid vs
  lexical-floor-only; land together vs split at the port boundary; the cosine
  threshold + calibration source; and the operator-gated re-baseline sequencing.
- Affected surfaces (implementation): clustering internals + a new port seam + host
  adapter. No report snapshot/teaser/ladder/pricing shape change; no DB; no network
  beyond the local CPU embedding model behind the port.
- Reviewer rules the implementation will trigger: R1 (requirements match --
  `extracted_*` change), R2 (failure-branch fixtures), R10 (maintainability), R12 (CI
  runs the enrolled test).

### Files touched
- `plans/PR-Deflection-Hybrid-Question-Clustering.md`

## Mechanism

`_question_subclusters` runs per topic/pain group, so each call sees a small set of
question gists -- exact pairwise work is affordable and no ANN/vector store is needed.
The lexical floor replaces candidate-by-banding with a prefix-filtered exact join:
dedupe identical gists, build a rare-token-first prefix, gate pairs by a length-ratio
bound, then verify with `_jaccard` against `_SUBCLUSTER_JACCARD_THRESHOLD`; union-find
keeps the current single-link grouping. This is exact (no probabilistic miss) and
order-insensitive because membership no longer depends on which row a band bucket saw
first.

The embedding booster runs only over pairs the floor left unmerged. Gists are embedded
once per group via the injected `EmbeddingPort`; exact `cosine` (not the shipped
ivfflat ANN, which is approximate) compares them; pairs above the calibrated threshold
are unioned. Determinism holds only if the model name + revision and an offline/cache
contract are pinned -- the shipped embedding service does not pin a revision today, so
this slice must add it (see the acceptance criteria) -- with the threshold fixed,
inference on CPU, and iteration order fixed. Auditability is preserved by
tagging each merge with its provenance: lexical merges keep their shared-token
evidence; embedding merges carry their cosine, so a reviewer can always see why two
questions were joined.

## Intentional
- Lexical floor always runs and is fully auditable; the package stays
  standalone/model-free when no port is injected (graceful degradation, mirroring the
  optional LLM/skill ports).
- Embedding is a host-injected port, not a package dependency -- the same seam as
  `HostLLMClient` / `HostSkillStore`; keeps the "no model in the loop" boundary a host
  decision, per the #1507 correction.
- Exact cosine over per-group gists (small N), not the ivfflat ANN index -- approx
  search would break "same data, same numbers".
- Every merge records provenance (shared tokens or cosine) so the booster never becomes
  an unexplainable black box.
- The lexical floor is the source of the published floor guarantee; the booster only
  ever adds recall, never invents a false repeat past verification.
- Determinism is contingent on a pinned model: the embedding path must pin model name +
  revision + offline/cache behavior, which the shipped service does not do today. This
  is an explicit acceptance criterion, not an assumption the floor/booster inherit.

## Deferred
- The implementation PR (follow-up). Proposed file set:
  `extracted_content_pipeline/ticket_faq_markdown.py`, a new
  `extracted_content_pipeline/embedding_port.py`,
  `atlas_brain/_content_ops_infrastructure.py`,
  `atlas_brain/_content_ops_services.py`,
  `tests/test_extracted_ticket_faq_markdown.py`, and a host-adapter wiring test
  `tests/test_content_ops_embedding_port.py` (port offline/pin behavior + bundle
  injection). Estimated ~400 code LOC; if over the 400 cap, split at the port boundary
  (Slice A floor + seam, Slice B booster + adapter + host-port tests).
- Operator-gated re-baseline: golden-fixture refresh + the live CFPB re-run + sign-off
  on the moved repeat counts before any paid artifact (#1504 sequencing).
- Cosine-threshold calibration data: extend the #1483 synthetic generator to emit
  controlled-similarity paraphrase pairs as the labeled set that fixes the threshold.
- Replacing the primary anchor/token_set clustering in `support_ticket_clustering.py`
  with embeddings (this targets only the question sub-clustering, #1504's locus).
- Persisted pgvector / Neo4j cross-report semantic dedup and similar-ticket retrieval.

Parked hardening: none.

## Verification
- This RFC PR: plan-shape, plan files-touched, plan-code-consistency, and the PR-body
  1b contract are green; no code changes, so the extracted suite is unaffected.
- The implementation PR will add cases in `tests/test_extracted_ticket_faq_markdown.py`
  (documented-bar parity, order-insensitivity, no-port byte-identity, stub-port
  reworded-pair merge with cosine, determinism, merge-provenance), add host-adapter
  wiring tests (port offline/pinned behavior + bundle injection), and run
  `scripts/run_extracted_pipeline_checks.sh` plus `scripts/check_ascii_python.sh`.

## Estimated diff size
| Area | Est LOC |
|---|---:|
| Plan doc (this RFC PR) | ~175 |
| Total | ~175 |

The implementation follow-up is estimated separately (~370 code LOC) in Deferred.
