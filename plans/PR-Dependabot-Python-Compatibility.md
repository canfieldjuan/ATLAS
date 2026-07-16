# PR-Dependabot-Python-Compatibility

## Why this slice exists

The operator authorized a focused compatibility/split-update migration after
Dependabot PR #2108 attempted 42 Python updates as one group and turned the
root/ASR dependency matrix red before tests could start. The unit gate proves
that its `torch==2.13.0` selection conflicts with its generated
`cuda-toolkit==13.3.1` pin: Torch requires CUDA toolkit 13.0.3. The dual-Python
root/ASR compiler independently rejects the selected boto3 artifact because
the frozen resolver cutoff predates it. CI review also proves that the current
Torchaudio release ceiling is 2.11, so advancing Torch would produce an
unsupported binary pair used by the real ASR server. FastAPI 0.139 additionally
represents included routers lazily, which made one host-wiring test assume an
obsolete flat router shape even though the mounted route remains reachable.
This is production hardening: the update cannot ship until the direct inputs,
generated closure, and compatibility test all agree.
The resulting diff exceeds the soft cap because one indivisible
compatibility proof must carry the generated dual-Python lock, the six existing
aggregate-router regressions, and the UI-root regression exposed by the same
FastAPI change; splitting those fixes would leave the selected dependency
release red or semantically unverified.

### Problem-derived contract

- Root cause: Dependabot directly changes generated root/ASR lock cells while
  combining releases that the real resolver cannot install together. Its Torch
  update conflicts with both the generated CUDA closure and the real ASR
  server's Torchaudio binary pair; the resolver does not encode that ABI
  compatibility rule. Its frozen upload cutoff also makes a selected direct
  boto3 version invisible to the dual-interpreter compiler. Separately,
  FastAPI 0.139 preserves included routers as lazy router objects, while six
  existing aggregate-router structural assertions assume every entry exposes
  `.path`. The existing UI-build root override also removes only top-level
  routes, leaving the Ollama root inside FastAPI 0.139's lazy include ahead of
  the intended browser UI handler.
- Correct fix must touch/change: Update only the selected root/ASR direct
  dependency inputs other than Torch and the resolver cutoff, regenerate
  `constraints.root-asr.txt` through `scripts/compile_root_asr_constraints.py`,
  retain the compiler's lock inclusion in both entrypoints and its digest
  binding in the root entrypoint, make the affected aggregate-router tests
  inspect direct and lazily included routers, and preserve the existing UI
  build root behavior without mutating FastAPI's route list. The result must
  solve for Python 3.10 and 3.11 without changing ASR's existing Torch input,
  browser UI response, or non-browser Ollama health response.
- Must not change: Do not modify product code beyond the exact existing UI-root
  behavior required by this FastAPI compatibility repair, runtime
  configuration, CI workflow behavior, or customer-facing surfaces. Do not
  edit generated lock cells by hand, force a Starlette pin to preserve a test implementation
  detail, change the established browser UI or non-browser Ollama root
  semantics, upgrade the separate `atlas_edge` NeMo dependency, or include the
  Thinc 9 migration tracked by issue #2100 / Dependabot PR #2093.

## Scope (this PR)

Ownership lane: dependency/dependabot
Slice phase: Production hardening

1. Split the root/ASR-compatible direct inputs from Dependabot PR #2108 into a
   human-owned compatibility update: FastAPI/Uvicorn, LangGraph/LangChain,
   Anthropic, and boto3. Keep `torch==2.12.1` because no matching Torchaudio
   release exists for the proposed Torch update.
2. Advance the compiler's fixed artifact cutoff from 2026-07-11 to
   `2026-07-14T19:30:00Z`, immediately after the latest selected artifact
   (boto3 1.43.48 at 19:29:57Z; botocore 1.43.48 at 19:29:45Z), then regenerate
   the canonical root/ASR lock and requirement digests.
3. Update only the six aggregate-router structural assertions that the unit
   gate proves fail under FastAPI's lazy included-router representation:
   Content Ops input-provider wiring, tenant B2B exposure, BYOK registration,
   LLM gateway registration, and Content Ops generated-assets host wiring.
   Preserve each test's existing route, auth, pool, provider, and negative
   route assertions.
4. Move the existing UI-build content negotiation from the top-level route-list
   mutation to the actual Ollama root handler, while leaving `main.py` to mount
   UI static assets. Add a real FastAPI entrypoint test that proves browsers
   receive the UI, non-browser clients receive the Ollama health response, and
   static assets remain reachable under FastAPI 0.139.
5. Prove the complete root/ASR lock graph through the real compiler, `--check`,
   its CI-enrolled contract tests, the Content Ops workflow suite under the
   selected FastAPI release, and target-Python dependency-resolution dry runs.
   Leave every other #2108 update for a separate owner-compatible slice.

### Review Contract

- Acceptance criteria:
  - [ ] The dual Python 3.10/3.11 compiler generates a single canonical lock
    from the changed source inputs without a solver error.
  - [ ] `requirements.txt` and `requirements.asr.txt` consume the regenerated
    lock, and the root entrypoint has the compiler-produced matching hash.
  - [ ] The root Torch input remains at 2.12.1; no new unsupported Torch /
    Torchaudio binary pair or generated CUDA change is introduced.
  - [ ] Each current aggregate-router assertion locates routes through both
    direct and FastAPI 0.139 lazy included-router representations without
    weakening its existing route, auth, pool, provider, or negative-route
    assertions.
  - [ ] With a UI build present, the actual FastAPI app returns the UI index at
    `/` for browser accepts, preserves the Ollama health body for non-browser
    accepts, and serves static assets without direct `app.routes` mutation.
  - [ ] The root and ASR entrypoints resolve under their generated lock for the
    supported target interpreter.
  - [ ] No Thinc, spaCy, `atlas_edge`, or unrelated product/runtime change is
    included.
- Reachability proof: `scripts/compile_root_asr_constraints.py` is the real
  lock-generation entrypoint; its observable output is the canonical lock plus
  a root digest binding, then a freshness check and the CI-enrolled constraints
  contract test verify that generated artifact. `tests/test_atlas_ui_root.py`
  mounts the real Ollama router and UI static surface; its observable browser,
  health, and asset responses prove the updated root wiring.
- Affected surfaces: root requirements, ASR requirements, the dual-Python
  lock compiler, generated root/ASR constraints, aggregate-router tests, the
  Ollama compatibility root, UI static mount, and their existing CI workflows.
- Risk areas: a stale or manually edited lock can make Docker/CI installs
  unsatisfiable; advancing the cutoff can choose an unintended transitive
  release; a resolver can accept ABI-incompatible binary packages; a FastAPI
  test can confuse private router representation with route reachability; an
  obsolete top-level route mutation can leave the UI root behind an included
  router; broadening into #2108's other package surfaces would make the compatibility
  proof meaningless.
- Reviewer rules triggered: R1, R2, R5, R10, R11, R12, R14.

### Files touched

- `HARDENING.md`
- `atlas_brain/api/ollama_compat.py`
- `atlas_brain/main.py`
- `constraints.root-asr.txt`
- `plans/PR-Dependabot-Python-Compatibility.md`
- `requirements.asr.txt`
- `requirements.txt`
- `scripts/compile_root_asr_constraints.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_atlas_content_ops_input_provider.py`
- `tests/test_atlas_ui_root.py`
- `tests/test_b2b_tenant_data_freshness.py`
- `tests/test_byok_keys.py`
- `tests/test_llm_gateway_router.py`

## Mechanism

The PR treats `requirements.txt` and `requirements.asr.txt` as dependency
inputs and `constraints.root-asr.txt` as a generated artifact. It updates only
the selected direct inputs that do not require an unavailable ABI-matched
companion package, moves `EXCLUDE_NEWER` to a verified cutoff that admits them,
and invokes the pinned `uv` compiler for both target Python versions. The
compiler writes one marker-aware union lock and rewrites the root entrypoint
with its SHA-256 binding while keeping both entrypoints bound to the lock. No
CUDA or NVIDIA cell is edited directly. Each affected test recursively inspects
a FastAPI lazy include's original router only to locate the same terminal route
objects it already asserts; it does not alter host routing or extract a shared
test utility outside this CI failure set. The existing root content negotiation
lives on the Ollama router's terminal route, which FastAPI dispatches before
the UI static mount; the main module no longer mutates the framework-owned
route list. The entrypoint test mounts that real router and static assets under
the updated FastAPI version to prove the existing browser and health responses.

## Intentional

- Do not carry all 42 #2108 updates: `atlas_edge` NeMo, content-ops fixture,
  graphiti, and video dependency changes lack this root/ASR compatibility
  proof and stay out of the slice.
- Do not advance Torch or pin CUDA by hand: Torchaudio 2.11 is the newest
  available release and the ASR server imports both packages, so a Torch /
  Torchaudio compatibility migration requires a separately validated pair.
- Do not pin Starlette solely to retain FastAPI's former flat router list; the
  actual mounted Content Ops route remains reachable and the structural test
  must represent both router forms instead.
- Do not change the root product contract: browser requests still receive the
  built UI while non-browser clients still receive `Ollama is running`; this
  repair moves the existing decision to the route FastAPI actually dispatches.
- Keep the existing Python 3.10 + 3.11 lock contract and pinned `uv` version;
  this is a closure repair, not a tooling migration.
- Do not treat a local Python 3.13 pip dry-run's missing Kokoro distribution as
  a root compatibility defect: the lock contract explicitly targets Python
  3.10 + 3.11, so resolver evidence uses a target Python rather than the
  builder's unsupported interpreter.

## Deferred

- Dependabot PR #2093 / issue #2100 remains the separate Thinc 9 + spaCy/NLP
  migration.
- `HARDENING.md` "ASR Torch/Torchaudio compatibility baseline" records the
  pre-existing unmatched pair for a dedicated ASR runtime validation slice;
  this PR only prevents a further unsupported Torch advancement.
- #2108's omitted packages remain for follow-up, split by their owning
  dependency surface after this root/ASR compatibility proof lands.
- Closing the original incompatible #2108 is a PR-management action after this
  replacement's verification; it is not evidence that every omitted package
  has been upgraded.

Parked hardening: `HARDENING.md` "ASR Torch/Torchaudio compatibility baseline".

## Verification

- Run the real compiler, freshness check, contract tests, and target-Python
  resolver dry runs:

    python scripts/compile_root_asr_constraints.py                 # pass
    python scripts/compile_root_asr_constraints.py --check         # pass
    python -m pytest tests/test_compile_root_asr_constraints.py -q # 10 passed
    /tmp/atlas-fastapi-compat-probe/bin/python -m pytest \
      tests/test_atlas_content_ops_generated_assets_api.py \
      tests/test_content_ops_brand_voice_profiles.py \
      tests/test_content_ops_brand_voice_profiles_api.py \
      tests/test_content_ops_zendesk_credentials.py \
      tests/test_content_ops_zendesk_export_api.py \
      tests/test_setup_content_ops_zendesk_credentials.py \
      tests/test_atlas_content_ops_input_provider.py::test_api_aggregator_wires_content_ops_input_provider \
      tests/test_atlas_content_ops_input_provider.py::test_api_preview_route_applies_support_ticket_input_provider \
      tests/test_atlas_content_ops_input_provider.py::test_api_plan_route_applies_support_ticket_input_provider \
      tests/test_b2b_tenant_data_freshness.py::test_api_router_exposes_only_tenant_b2b_paths \
      tests/test_byok_keys.py::test_byok_router_registered_in_aggregator \
      tests/test_llm_gateway_router.py::test_router_registered_in_api_aggregator \
      tests/test_atlas_ui_root.py -q  # 87 passed after root repair
    python -m pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements.txt      # pass
    python -m pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements.asr.txt  # pass

- Guarded `scripts/push_pr.sh` local review immediately before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 17 |
| `atlas_brain/api/ollama_compat.py` | 13 |
| `atlas_brain/main.py` | 22 |
| `constraints.root-asr.txt` | 28 |
| `plans/PR-Dependabot-Python-Compatibility.md` | 237 |
| `requirements.asr.txt` | 2 |
| `requirements.txt` | 14 |
| `scripts/compile_root_asr_constraints.py` | 2 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 19 |
| `tests/test_atlas_content_ops_input_provider.py` | 17 |
| `tests/test_atlas_ui_root.py` | 34 |
| `tests/test_b2b_tenant_data_freshness.py` | 12 |
| `tests/test_byok_keys.py` | 14 |
| `tests/test_llm_gateway_router.py` | 14 |
| **Total** | **445** |
