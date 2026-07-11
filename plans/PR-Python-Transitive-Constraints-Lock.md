# PR-Python-Transitive-Constraints-Lock

## Why this slice exists

Issue #2040 remains open after #2072 removed the duplicate root NeMo owner:
the root plus standalone-ASR environment still pins only direct dependencies,
so an unchanged repository can resolve a different transitive graph tomorrow.
That is a real reproducibility risk identified by #2035/#2036, and the
aggregate unit gate that was required before constraining the full graph now
exists (#2041). Production hardening is therefore appropriate.

Current-code and resolver reconstruction narrows "full tree" to the shared
root/standalone-ASR x86_64 Linux environment. Root requirements are consumed
under Python 3.11 by the aggregate gate and most product checks, and under
Python 3.10 by extracted-package checks. Real uv 0.10.10 resolutions at
`origin/main` produced 281 packages on 3.11 and 285 on 3.10. The graphs differ
in four 3.10-only compatibility packages and nine version/presence cells, so a
single unmarked lock would be false. Atlas Edge is a separate aarch64/Jetson
deployment and Content Ops CI has a conflicting direct LangGraph pin; neither
can truthfully share this lock.

This PR will exceed the 400-LOC soft cap: the generated marker-aware lock is
approximately 295 installable lines before its header, and a trustworthy lock
also needs the small deterministic compiler, contract tests, CI regeneration
check, plan, and #2072 teardown. Splitting generated data from its generator or
enforcement would create an unauditable or unused lock, so the overage is
indivisible while changed executable symbols remain small.

### Problem-derived contract

- Root cause: `requirements.txt` and `requirements.asr.txt` constrain direct
  dependencies but let pip choose every transitive version at install time.
  Python 3.10 and 3.11 do not resolve identical valid graphs, and no committed
  artifact or CI check records and enforces those interpreter-specific choices.
- Correct fix must touch/change: generate one deterministic root/ASR constraints
  artifact from real Python 3.10 and 3.11 x86_64 Linux resolutions; represent
  common pins once and divergent cells with explicit `python_version` markers;
  make both requirements files consume it; bind its digest into
  `requirements.txt` so a lock-only update cannot evade the existing consumer
  fleet; copy the referenced artifact into the root image before pip runs; add
  a regeneration/contract check; enroll that check in CI; archive this
  session's merged #2072 plan and refresh the plan index.
- Must not change: direct dependency names, versions, extras, or order beyond
  the constraint include/digest; the NeMo git SHA; runtime, Docker, ASR, voice,
  or product behavior; Docker build/runtime behavior beyond copying the new
  pip input beside `requirements.txt`; `requirements.content_ops_ci.txt`;
  Atlas Edge or other subproject requirements; package indexes; GPU/CPU backend selection;
  Resolution Audit/Content Ops lanes; or buyer-visible product shape.

## Scope (this PR)

Ownership lane: dependency-hardening/python-requirements
Slice phase: Production hardening

1. Commit and consume a marker-aware transitive lock for the combined root and
   standalone-ASR environment on Python 3.10/3.11 x86_64 Linux.
2. Add a deterministic compiler and contract tests that fail on stale digest,
   missing includes, unpinned cells, marker gaps, or regeneration drift.
3. Add a focused CI check that regenerates the lock from both interpreter
   graphs; rely on the digest change in `requirements.txt` to trigger the
   existing real install/test fleet whenever the lock changes.
4. Archive only the merged `PR-NeMo-Requirements-Dedup.md` plan and rebuild the
   plan index.
5. Preserve the root container build by copying the constraints artifact into
   `/app` beside the requirement file before pip resolves its include.

Max files: 10

### Review Contract

- Acceptance criteria:
  - [ ] A single constraints artifact contains every resolved root/ASR package
        exactly once when common or in complete, non-overlapping 3.10/3.11
        marker branches when divergent.
  - [ ] The artifact retains NeMo's exact git SHA and uses the repository's
        existing default package-index behavior without selecting a new Torch
        backend.
  - [ ] Both root and ASR requirement files include the constraints artifact,
        while all pre-existing direct requirement text/order remains unchanged.
  - [ ] `requirements.txt` records the generated lock digest, and the contract
        test rejects a changed lock unless that digest is refreshed.
  - [ ] The root Docker build copies the constraints artifact before its
        existing pip install step reads `requirements.txt`.
  - [ ] Regeneration with uv 0.10.10 for Python 3.10 and 3.11 x86_64 Linux is
        byte-identical to the committed artifact.
  - [ ] Existing root-requirements consumers are re-triggered by the digest
        line, and focused 3.10/3.11 resolver plus aggregate-gate evidence is
        green.
  - [ ] Only this session's merged #2072 plan is archived.
- Reachability proof: run the compiler against the real requirement files and
  both supported interpreter targets, then let the changed `requirements.txt`
  flow through the existing Python 3.10 and 3.11 install/test consumers. The
  observable results are byte-identical regeneration, complete marker coverage,
  successful constrained resolution, and green aggregate/extracted gates.
- Affected surfaces: root Python installs, standalone ASR installs, root image
  dependency build, Python 3.10/3.11 CI resolution, scheduled SCA input, and plan teardown.
- Risk areas: incomplete marker partitioning, a stale lock accepted by CI,
  accidentally changing package indexes/Torch backend, VCS-reference drift,
  or a lock change that does not trigger real consumers.
- Reviewer rules triggered: R1, R2, R11, R12, R14.

### Files touched

- `.github/workflows/python_constraints_checks.yml`
- `Dockerfile`
- `constraints.root-asr.txt`
- `plans/INDEX.md`
- `plans/PR-Python-Transitive-Constraints-Lock.md`
- `plans/archive/PR-NeMo-Requirements-Dedup.md`
- `requirements.asr.txt`
- `requirements.txt`
- `scripts/compile_root_asr_constraints.py`
- `tests/test_compile_root_asr_constraints.py`

## Mechanism

The compiler removes only this repository's constraint include from temporary
copies of the two direct requirement inputs, then invokes pinned uv semantics
twice: Python 3.10 and Python 3.11 on x86_64 manylinux. It parses each concrete
`name==version` or direct-reference result by normalized project name. Identical
cells emit once; packages or versions that differ emit two mutually exclusive
`python_version < '3.11'` / `python_version >= '3.11'` branches. A stable header
records the tool version and regeneration command.

After writing the artifact, the compiler hashes its exact bytes and updates the
single digest comment beside the `-c` include in `requirements.txt`.
`requirements.asr.txt` consumes the same artifact without owning the digest.
`--check` regenerates in a temporary directory and compares both the artifact
and digest without editing the tree. Focused tests exercise the merge algorithm
with synthetic common, divergent, one-sided, direct-reference, and malformed
graphs, then audit the real files and root Docker copy boundary. The Dockerfile
copies both pip inputs in one layer without changing its install command. A
dedicated workflow installs uv 0.10.10 and runs `--check`; the digest line makes
every accepted artifact update also touch the already-enrolled root requirements path.

## Intentional

- One marker-aware artifact is used instead of two near-duplicate 281/285-line
  locks. The representation makes interpreter divergence reviewable and avoids
  two authorities for common pins.
- uv resolves with the repository's default index behavior. CPU-only or CUDA-
  specific `--torch-backend` flags are rejected because existing pip consumers
  do not select those indexes and a lock must describe the deployed resolver.
- Local development remains explicit: root-only installs constrain the root
  graph; ASR auto-start still requires installing `requirements.asr.txt`, as
  documented and reviewed on #2072.
- The lock is version-pinned but not `--require-hashes`. Cross-interpreter,
  platform-specific wheels plus a VCS NeMo source make hash enforcement a
  separate supply-chain decision rather than a safe incidental addition.

## Deferred

- #2040C: pin and lock the independent Atlas Edge aarch64/Jetson dependency
  graph, including its currently unpinned `nemo_toolkit[asr]`. The reviewer
  explicitly surfaced this on #2072 and it is recorded on issue #2040.
- A Content Ops CI transitive lock remains separate because that Python 3.11
  surface directly pins `langgraph==1.2.8` while root pins `1.2.7`; silently
  forcing either authority into the other would change a proven test surface.
- Other standalone Python subprojects and `--require-hashes` remain separate
  deployment/supply-chain slices.

Parked hardening: none; every discovered adjacent dependency surface is tracked
on reopened issue #2040.

## Verification

- Pre-plan resolver probe: uv 0.10.10 resolved the combined root/ASR graph to
  285 packages for Python 3.10 and 281 for Python 3.11 on x86_64 manylinux.
- Pre-plan divergence probe: four packages are 3.10-only; nine package cells
  differ by presence or version, including NumPy, Pandas, SciPy, NetworkX, and
  ONNX Runtime.
- Test-first run before the compiler existed: collection failed as expected on
  the missing compiler module. After the compiler existed but before generation:
  5 passed and the real-file test failed on the missing lock artifact.
- `python -m pytest tests/test_compile_root_asr_constraints.py
  tests/test_nemo_requirements_ownership.py
  tests/test_atlas_main_voice_startup.py
  tests/test_content_ops_ci_requirements_workflows.py
  tests/test_check_unit_gate.py -q` -- 61 passed, with one pre-existing Torch
  `pynvml` deprecation warning.
- `python scripts/compile_root_asr_constraints.py --check` -- passed; committed
  digest `2dcc58c13e8a6ccd3a01d004e4a7b3b65834af4a7b3b0fd24026601f8941e314`.
- uv 0.10.10 concrete compile with the committed constraints on Python 3.10
  and 3.11 -- 285 and 281 packages; both outputs were byte-identical to their
  respective pre-lock resolver probes.
- Python YAML parse of `.github/workflows/python_constraints_checks.yml` --
  passed.
- Root Docker copy-order regression is included in the 61-test focused run;
  it proves the constraints artifact is present before pip reads the root file.
- `docker build --check .` -- passed with no warnings.
- `git diff --check` -- passed.
- Plan sync/check, plan/code consistency, ASCII policy, and exact direct-
  requirement diff audit -- passed.
- Pending at push: guarded pre-push local review.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/python_constraints_checks.yml` | 39 |
| `Dockerfile` | 4 |
| `constraints.root-asr.txt` | 297 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Python-Transitive-Constraints-Lock.md` | 210 |
| `plans/archive/PR-NeMo-Requirements-Dedup.md` | 0 |
| `requirements.asr.txt` | 2 |
| `requirements.txt` | 3 |
| `scripts/compile_root_asr_constraints.py` | 173 |
| `tests/test_compile_root_asr_constraints.py` | 131 |
| **Total** | **862** |
