# PR-Dependabot-Python-Compatibility

## Why this slice exists

The operator authorized a focused compatibility/split-update migration after
Dependabot PR #2108 attempted 42 Python updates as one group and turned the
root/ASR dependency matrix red before tests could start. The unit gate proves
that its `torch==2.13.0` selection conflicts with its generated
`cuda-toolkit==13.3.1` pin: Torch requires CUDA toolkit 13.0.3. The dual-Python
root/ASR compiler independently rejects the selected boto3 artifact because
the frozen resolver cutoff predates it. This is production hardening: the
dependency update cannot be installed safely until the real lock compiler
produces one coherent Python 3.10/3.11 closure.

### Problem-derived contract

- Root cause: Dependabot directly changes generated root/ASR lock cells while
  combining releases that the real resolver cannot install together. The root
  inputs select `torch==2.13.0`, but the generated CUDA closure is pinned to a
  different toolkit release; its frozen upload cutoff also makes a selected
  direct boto3 version invisible to the dual-interpreter compiler.
- Correct fix must touch/change: Update only the selected root/ASR direct
  dependency inputs and the resolver cutoff, regenerate
  `constraints.root-asr.txt` through `scripts/compile_root_asr_constraints.py`,
  retain the compiler's lock inclusion in both entrypoints and its digest
  binding in the root entrypoint. The result must solve for Python 3.10 and
  3.11 and let the resolver—not a hand edit—choose Torch's CUDA transitive
  closure.
- Must not change: Do not modify product code, runtime configuration, CI
  workflow behavior, or customer-facing surfaces. Do not edit generated lock
  cells by hand, upgrade the separate `atlas_edge` NeMo dependency, or include
  the Thinc 9 migration tracked by issue #2100 / Dependabot PR #2093.

## Scope (this PR)

Ownership lane: dependency/dependabot
Slice phase: Production hardening

1. Split the root/ASR-compatible direct inputs from Dependabot PR #2108 into a
   human-owned compatibility update: FastAPI/Uvicorn, LangGraph/LangChain,
   Anthropic, boto3, and the Torch input only if the dual-Python resolver
   selects a coherent CUDA closure.
2. Advance the compiler's fixed artifact cutoff from 2026-07-11 to
   `2026-07-14T19:30:00Z`, immediately after the latest selected artifact
   (boto3 1.43.48 at 19:29:57Z; botocore 1.43.48 at 19:29:45Z), then regenerate
   the canonical root/ASR lock and requirement digests.
3. Prove the complete root/ASR lock graph through the real compiler, `--check`,
   its CI-enrolled contract tests, and target-Python dependency-resolution dry
   runs. Leave every other #2108 update for a separate owner-compatible slice.

### Review Contract

- Acceptance criteria:
  - [ ] The dual Python 3.10/3.11 compiler generates a single canonical lock
    from the changed source inputs without a solver error.
  - [ ] `requirements.txt` and `requirements.asr.txt` consume the regenerated
    lock, and the root entrypoint has the compiler-produced matching hash.
  - [ ] The selected Torch update does not preserve Dependabot's incompatible
    CUDA 13.3.1 generated constraint; the resolver chooses the closure.
  - [ ] The root and ASR entrypoints resolve under their generated lock for the
    supported target interpreter.
  - [ ] No Thinc, spaCy, `atlas_edge`, or product/runtime change is included.
- Reachability proof: `scripts/compile_root_asr_constraints.py` is the real
  lock-generation entrypoint; its observable output is the canonical lock plus
  a root digest binding, then a freshness check and the CI-enrolled constraints
  contract test verify that generated artifact.
- Affected surfaces: root requirements, ASR requirements, the dual-Python
  lock compiler, generated root/ASR constraints, and the existing Python
  Constraints Checks workflow.
- Risk areas: a stale or manually edited lock can make Docker/CI installs
  unsatisfiable; advancing the cutoff can choose an unintended transitive
  release; a Torch input update can silently select an incompatible CUDA
  closure; broadening into #2108's other package surfaces would make the
  compatibility proof meaningless.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R14.

### Files touched

- `constraints.root-asr.txt`
- `plans/PR-Dependabot-Python-Compatibility.md`
- `requirements.asr.txt`
- `requirements.txt`
- `scripts/compile_root_asr_constraints.py`

## Mechanism

The PR treats `requirements.txt` and `requirements.asr.txt` as dependency
inputs and `constraints.root-asr.txt` as a generated artifact. It updates only
the selected direct inputs, moves `EXCLUDE_NEWER` to a verified cutoff that
admits them, and invokes the pinned `uv` compiler for both target Python
versions. The compiler writes one marker-aware union lock and rewrites the
root entrypoint with its SHA-256 binding while keeping both entrypoints bound
to the lock. No CUDA or NVIDIA cell is edited directly: the generated closure
is the compatibility decision.

## Intentional

- Do not carry all 42 #2108 updates: `atlas_edge` NeMo, content-ops fixture,
  graphiti, and video dependency changes lack this root/ASR compatibility
  proof and stay out of the slice.
- Do not pin `cuda-toolkit==13.0.3` by hand: a trailing-zero package spelling
  and every NVIDIA transitive pin belong to the resolver output.
- Keep the existing Python 3.10 + 3.11 lock contract and pinned `uv` version;
  this is a closure repair, not a tooling migration.
- Do not treat a local Python 3.13 pip dry-run's missing Kokoro distribution as
  a root compatibility defect: the lock contract explicitly targets Python
  3.10 + 3.11, so resolver evidence uses a target Python rather than the
  builder's unsupported interpreter.

## Deferred

- Dependabot PR #2093 / issue #2100 remains the separate Thinc 9 + spaCy/NLP
  migration.
- #2108's omitted packages remain for follow-up, split by their owning
  dependency surface after this root/ASR compatibility proof lands.
- Closing the original incompatible #2108 is a PR-management action after this
  replacement's verification; it is not evidence that every omitted package
  has been upgraded.

Parked hardening: none.

## Verification

- Run the real compiler, freshness check, contract tests, and target-Python
  resolver dry runs:

    python scripts/compile_root_asr_constraints.py
    python scripts/compile_root_asr_constraints.py --check
    python -m pytest tests/test_compile_root_asr_constraints.py -q
    python -m pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements.txt
    python -m pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements.asr.txt

- Guarded `scripts/push_pr.sh` local review immediately before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `constraints.root-asr.txt` | 36 |
| `plans/PR-Dependabot-Python-Compatibility.md` | 144 |
| `requirements.asr.txt` | 2 |
| `requirements.txt` | 16 |
| `scripts/compile_root_asr_constraints.py` | 2 |
| **Total** | **200** |
