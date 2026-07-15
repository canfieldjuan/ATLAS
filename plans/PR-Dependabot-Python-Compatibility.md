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

### Problem-derived contract

- Root cause: Dependabot directly changes generated root/ASR lock cells while
  combining releases that the real resolver cannot install together. Its Torch
  update conflicts with both the generated CUDA closure and the real ASR
  server's Torchaudio binary pair; the resolver does not encode that ABI
  compatibility rule. Its frozen upload cutoff also makes a selected direct
  boto3 version invisible to the dual-interpreter compiler. Separately,
  FastAPI 0.139 preserves included routers as lazy router objects, while an
  existing structural test assumes every aggregate entry exposes `.path`.
- Correct fix must touch/change: Update only the selected root/ASR direct
  dependency inputs other than Torch and the resolver cutoff, regenerate
  `constraints.root-asr.txt` through `scripts/compile_root_asr_constraints.py`,
  retain the compiler's lock inclusion in both entrypoints and its digest
  binding in the root entrypoint, and make the affected host-wiring test
  inspect direct and lazily included routers. The result must solve for Python
  3.10 and 3.11 without changing ASR's existing Torch input or product route
  behavior.
- Must not change: Do not modify product code, runtime configuration, CI
  workflow behavior, or customer-facing surfaces. Do not edit generated lock
  cells by hand, force a Starlette pin to preserve a test implementation
  detail, upgrade the separate `atlas_edge` NeMo dependency, or include the
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
3. Update only the Content Ops host-wiring test traversal for FastAPI's lazy
   included-router representation, preserving its existing route, auth, pool,
   and provider assertions.
4. Prove the complete root/ASR lock graph through the real compiler, `--check`,
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
  - [ ] The Content Ops host-wiring test locates routes through both direct and
    FastAPI 0.139 lazy included-router representations without weakening its
    existing route, auth, pool, or provider assertions.
  - [ ] The root and ASR entrypoints resolve under their generated lock for the
    supported target interpreter.
  - [ ] No Thinc, spaCy, `atlas_edge`, or product/runtime change is included.
- Reachability proof: `scripts/compile_root_asr_constraints.py` is the real
  lock-generation entrypoint; its observable output is the canonical lock plus
  a root digest binding, then a freshness check and the CI-enrolled constraints
  contract test verify that generated artifact.
- Affected surfaces: root requirements, ASR requirements, the dual-Python
  lock compiler, generated root/ASR constraints, the Content Ops host-wiring
  test, and their existing CI workflows.
- Risk areas: a stale or manually edited lock can make Docker/CI installs
  unsatisfiable; advancing the cutoff can choose an unintended transitive
  release; a resolver can accept ABI-incompatible binary packages; a FastAPI
  test can confuse private router representation with route reachability;
  broadening into #2108's other package surfaces would make the compatibility
  proof meaningless.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R14.

### Files touched

- `HARDENING.md`
- `constraints.root-asr.txt`
- `plans/PR-Dependabot-Python-Compatibility.md`
- `requirements.asr.txt`
- `requirements.txt`
- `scripts/compile_root_asr_constraints.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`

## Mechanism

The PR treats `requirements.txt` and `requirements.asr.txt` as dependency
inputs and `constraints.root-asr.txt` as a generated artifact. It updates only
the selected direct inputs that do not require an unavailable ABI-matched
companion package, moves `EXCLUDE_NEWER` to a verified cutoff that admits them,
and invokes the pinned `uv` compiler for both target Python versions. The
compiler writes one marker-aware union lock and rewrites the root entrypoint
with its SHA-256 binding while keeping both entrypoints bound to the lock. No
CUDA or NVIDIA cell is edited directly. The Content Ops test recursively
inspects a FastAPI lazy include's original router only to locate the same route
objects it already asserts; it does not alter host routing.

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

    python scripts/compile_root_asr_constraints.py
    python scripts/compile_root_asr_constraints.py --check
    python -m pytest tests/test_compile_root_asr_constraints.py -q
    python -m pytest tests/test_atlas_content_ops_generated_assets_api.py tests/test_content_ops_brand_voice_profiles.py tests/test_content_ops_brand_voice_profiles_api.py tests/test_content_ops_zendesk_credentials.py tests/test_content_ops_zendesk_export_api.py tests/test_setup_content_ops_zendesk_credentials.py -q
    python -m pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements.txt
    python -m pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements.asr.txt

- Guarded `scripts/push_pr.sh` local review immediately before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 17 |
| `constraints.root-asr.txt` | 28 |
| `plans/PR-Dependabot-Python-Compatibility.md` | 175 |
| `requirements.asr.txt` | 2 |
| `requirements.txt` | 14 |
| `scripts/compile_root_asr_constraints.py` | 2 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 19 |
| **Total** | **257** |
