# PR-Deflection-Embedding-Host-Adapter

## Why this slice exists

#1504 is now past the lexical floor and extracted booster-core slices. PR #1531
made the documented exact-Jaccard floor literal, and PR #1536 landed the
model-free embedding booster core behind an optional `EmbeddingPort`.

The remaining vertical gap is host activation wiring: the production Content
Ops execution route still builds FAQ/deflection services without an embedding
port, so the booster is landed but unreachable outside extracted-package tests.
The #1504 calibration note and the RFC plan require the host side to use the
pinned offline `mixedbread-ai/mxbai-embed-large-v1` model rather than the older
default MiniLM embedding service.

This slice wires only that host seam. It deliberately avoids the FAQ clustering
and report files that landed through #1538.

The slice lands above the 400 LOC soft target because the host seam is only
useful if the adapter, service-bundle injection, route gate, CI enrollment, and
tests proving those pieces ship together.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Vertical slice

1. Add a host `EmbeddingPort` adapter in
   `atlas_brain/_content_ops_infrastructure.py` that wraps the existing
   sentence-transformer embedding service and returns plain Python float
   vectors.
2. Extend the host sentence-transformer service to accept optional model
   revision and offline-cache flags while preserving existing defaults.
3. Add a pinned Content Ops FAQ embedding-port factory for
   `mixedbread-ai/mxbai-embed-large-v1` at cached revision
   `b33106f585b9ce46904ad7443a3b52b7a63e231c`, CPU device, and local-files-only
   loading.
4. Thread an optional FAQ embedding-port factory through the host execution
   services bundle and inject it into FAQ Markdown and deflection-report services
   when provided.
5. Wire the authenticated Content Ops route to pass a default-off gated factory,
   so the hosted deflection path can exercise the booster core only after the
   operator enables `ATLAS_CONTENT_OPS_FAQ_EMBEDDING_BOOSTER_ENABLED`.
6. Add host-side tests proving adapter conversion, pinned/offline factory
   construction, bundle injection, hidden-FAQ report reuse, no-port default
   behavior, and the default-off route gate.
7. Enroll the new atlas-brain-importing infrastructure test and the source
   files it covers in the dedicated atlas Content Ops service workflow.
8. Guard the heavy host infrastructure test with `pytest.importorskip("torch")`
   so torch-less extracted CI skips it during collection while the atlas
   workflow still runs it.

### Review Contract

- Acceptance criteria:
  - [ ] Host adapter converts batch embeddings from array/list-like service
        output into plain float vectors accepted by the extracted port.
  - [ ] The pinned factory constructs the mxbai service with model name,
        revision, CPU device, and local-files-only loading, without loading the
        model during service-bundle construction.
  - [ ] The service bundle keeps no-port behavior unchanged unless a caller
        provides a FAQ embedding-port factory.
  - [ ] When a factory is provided, both `faq_markdown` and
        `faq_deflection_report` use the same configured FAQ service, including
        when `faq_markdown` is hidden from customer-executable outputs.
  - [ ] The authenticated Content Ops route passes a gated factory into the
        execution-services provider, and the gate defaults off.
  - [ ] No extracted package imports host embedding code; the MCP/execution
        transport remains a thin service wrapper.
- Affected surfaces: host Content Ops infrastructure, host execution-services
  bundle, Content Ops route provider wiring, config flag, sentence-transformer
  service construction, host tests, atlas workflow enrollment, and this plan.
- Risk areas: accidental real-model load in tests or service construction,
  silently falling back to MiniLM, premature production activation before live
  mxbai calibration, and collision with open PR #1540.
- Reviewer rules triggered: R1, R2, R5, R10, R11, R12, R14.

### Files touched

- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `atlas_brain/_content_ops_infrastructure.py`
- `atlas_brain/_content_ops_services.py`
- `atlas_brain/api/__init__.py`
- `atlas_brain/config.py`
- `atlas_brain/services/embedding/sentence_transformer.py`
- `plans/PR-Deflection-Embedding-Host-Adapter.md`
- `tests/test_atlas_content_ops_execution_services.py`
- `tests/test_atlas_content_ops_infrastructure.py`

## Mechanism

The extracted package already owns the pure `EmbeddingPort` protocol. The host
adapter implements that protocol over an injected embedding service with an
`embed_batch` method, normalizing service output into immutable Python tuples of
floats. The adapter does not import or call the FAQ builder directly.

The pinned factory lives beside the existing host LLM/skill adapters. Production
construction uses `mixedbread-ai/mxbai-embed-large-v1`, revision
`b33106f585b9ce46904ad7443a3b52b7a63e231c`, `device="cpu"`, and
`local_files_only=True`. Construction remains lazy with respect to model load:
the model loads only if FAQ clustering actually asks the port for embeddings,
and the extracted booster still fails closed if embedding generation raises.

The execution-services bundle gets a `faq_embedding_port_factory` dependency
injection parameter. By default it is absent, preserving today’s no-port
behavior for tests, scripts, and dependency-light callers. The authenticated
Content Ops API route passes a gated helper that returns `None` unless
`ATLAS_CONTENT_OPS_FAQ_EMBEDDING_BOOSTER_ENABLED=true`, which keeps production
activation explicit until the mxbai live baseline is accepted.

## Intentional

- This PR does not change `extracted_content_pipeline/ticket_faq_markdown.py` or
  `extracted_content_pipeline/faq_deflection_report.py`; #1538 already handled
  the report surface, and #1536 already landed the extracted core.
- The default service-bundle factory remains no-port unless a caller explicitly
  passes a FAQ embedding-port factory. That keeps unit tests and standalone
  scripts from loading mxbai accidentally while the hosted route is wired behind
  a default-off operator flag.
- The factory pins a local cached revision instead of resolving the floating
  Hugging Face `main` ref at runtime. That is the determinism/offline contract
  from #1504/#1515.
- The hosted route gate defaults off because the mxbai thresholds still need
  live CFPB re-baseline evidence before production activation.

## Deferred

- Live CFPB re-baseline after the hosted route has the embedding port wired and
  the operator approves enabling the route flag for the live input.
- Surfacing semantic merge provenance in the report UI/output, if product wants
  buyer-visible explanation for embedding-driven joins.
- Multi-row lexical component semantic expansion remains deferred from #1536
  until a safer cluster-level criterion is proven.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile atlas_brain/config.py atlas_brain/_content_ops_infrastructure.py atlas_brain/_content_ops_services.py atlas_brain/api/__init__.py atlas_brain/services/embedding/sentence_transformer.py tests/test_atlas_content_ops_infrastructure.py tests/test_atlas_content_ops_execution_services.py.
- Command passed: pytest tests/test_atlas_content_ops_infrastructure.py::test_host_embedding_port_converts_batch_output_to_float_vectors tests/test_atlas_content_ops_infrastructure.py::test_content_ops_faq_embedding_factory_pins_mxbai_offline_model tests/test_atlas_content_ops_infrastructure.py::test_sentence_transformer_embedding_load_uses_revision_and_offline_flags tests/test_atlas_content_ops_execution_services.py::test_faq_markdown_uses_injected_host_embedding_port tests/test_atlas_content_ops_execution_services.py::test_faq_markdown_can_be_hidden_while_deflection_report_runs tests/test_atlas_content_ops_execution_services.py::test_faq_markdown_default_bundle_keeps_no_port_behavior tests/test_atlas_content_ops_execution_services.py::test_content_ops_api_route_gates_pinned_faq_embedding_factory -q - 7 passed, 1 warning.
- Command passed: pytest tests/test_atlas_content_ops_infrastructure.py -q - 15 passed, 1 warning.
- Command passed: pytest tests/test_atlas_content_ops_execution_services.py -q - 33 passed.
- Command passed: pytest tests/test_atlas_content_ops_infrastructure.py tests/test_atlas_content_ops_execution_services.py -q - 48 passed.
- Command passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main . - OK: 178 matching tests are enrolled.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh - 4109 passed, 10 skipped, 1 warning.
- Command passed: python scripts/sync_pr_plan.py plans/PR-Deflection-Embedding-Host-Adapter.md origin/main --check.
- Command passed: bash scripts/push_pr.sh tmp/deflection_embedding_host_adapter_pr_body.md -u origin claude/pr-deflection-embedding-host-adapter - local PR review passed and branch pushed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_input_provider_checks.yml` | 11 |
| `atlas_brain/_content_ops_infrastructure.py` | 62 |
| `atlas_brain/_content_ops_services.py` | 46 |
| `atlas_brain/api/__init__.py` | 9 |
| `atlas_brain/config.py` | 8 |
| `atlas_brain/services/embedding/sentence_transformer.py` | 15 |
| `plans/PR-Deflection-Embedding-Host-Adapter.md` | 162 |
| `tests/test_atlas_content_ops_execution_services.py` | 180 |
| `tests/test_atlas_content_ops_infrastructure.py` | 103 |
| **Total** | **596** |
