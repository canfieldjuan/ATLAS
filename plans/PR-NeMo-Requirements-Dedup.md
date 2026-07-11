# PR-NeMo-Requirements-Dedup

## Why this slice exists

Issue #2040 carries the one dependency left deliberately unpinned by #2036:
`requirements.txt` names the PyPI `nemo_toolkit[asr]` distribution while the
standalone ASR surface names a commit-pinned NeMo build in
`requirements.asr.txt`. The split ownership makes ordinary root installs
resolve a moving NeMo release and makes the full security sweep present two
sources for the same distribution. This is a reproducibility and deployment
risk found by the Phase-1 CI enforcement audit (#2035), so Production
hardening is appropriate after the aggregate unit gate landed in #2041.

Current-code reconstruction narrows the old issue text: the PR-time security
workflow audits requirement files separately, but the scheduled
`security_full_sweep.yml` still passes root and ASR files together. The real
NeMo runtime importer is the standalone `asr_server.py`, whose module and
operator instructions already require `requirements.asr.txt`; Atlas Edge owns
its separate deploy requirements. `atlas_brain.main` may spawn the standalone
server, but it does not import NeMo and the documented ASR setup remains a
separate install step.

### Problem-derived contract

- Root cause: one optional ASR distribution has two package-source owners: an
  unpinned PyPI declaration in the general root environment and a commit-pinned
  direct reference in the ASR environment. The root declaration is not a
  required dependency of the root application image or any root-only runtime
  import, so it both defeats #2036's reproducibility contract and duplicates
  the standalone ASR dependency boundary.
- Correct fix must touch/change: remove only the root NeMo declaration and its
  obsolete carve-out comment; retain the exact ASR git SHA; add a repository
  contract test proving NeMo has one canonical ASR declaration, is absent from
  root requirements, and remains covered by the joint full-sweep command; run
  the real joint pip resolver in no-install mode; archive this session's merged
  #2071 plan and refresh the plan index.
- Must not change: any other dependency name/version/order, the NeMo git SHA,
  `requirements.asr.txt`, Atlas Edge requirements, ASR/voice runtime code or
  defaults, Docker/runtime entrypoints, workflow behavior, buyer-visible
  product shape, #2037/#1737 files, or the separate transitive-lock design.

## Scope (this PR)

Ownership lane: dependency-hardening/python-requirements
Slice phase: Production hardening

1. Make `requirements.asr.txt` the sole root-repository owner of the
   standalone NeMo ASR dependency by removing its duplicate general-runtime
   declaration.
2. Add a static ownership/audit regression and exercise pip's actual joint
   requirement resolver without installing or upgrading dependencies.
3. Archive only `PR-Resolution-Audit-S6A2-Authoritative-Source-Admission.md`,
   the merged plan owned by this session, and rebuild `plans/INDEX.md`.

Max files: 5

### Review Contract

- Acceptance criteria:
  - [ ] `requirements.txt` contains no `nemo_toolkit` distribution and every
        remaining installable line keeps its existing text, order, and pin.
  - [ ] `requirements.asr.txt` retains exactly one `nemo_toolkit[asr]` direct
        reference at commit `0f378e9d8dd72630c911025b555f18658d44cc8f`.
  - [ ] The scheduled full security sweep still jointly audits
        `requirements.txt` and `requirements.asr.txt`.
  - [ ] The standalone ASR server still names `requirements.asr.txt` as its
        missing-dependency recovery path.
  - [ ] Under the CI Python 3.11 interpreter, pip parses the combined root plus
        ASR requirement set and resolves NeMo to the exact canonical SHA with
        no duplicate-source conflict; no package is installed locally.
  - [ ] Existing general-runtime, voice-startup, requirements-shape, and unit-
        gate tests remain green or within the committed ratchet.
  - [ ] Only this session's merged #2071 plan is archived.
- Reachability proof: invoke pip against both requirement files under Python
  3.11 with `--dry-run --no-deps` through NeMo SHA resolution, then let the
  PR's `requirements.txt` path trigger the consuming CI fleet; observable
  results are one canonical NeMo source and green install/test jobs using the
  reduced general dependency set. The local dry-run may be stopped after NeMo
  resolution because pip still downloads the unrelated 532 MB Torch wheel in
  dry-run mode.
- Affected surfaces: root Python dependency installation, standalone ASR
  dependency ownership, scheduled joint SCA input, and plan teardown.
- Risk areas: an overlooked root-only NeMo importer, accidental SHA movement,
  an unrelated dependency change, or reduced CI enrollment.
- Reviewer rules triggered: R1, R2, R11, R12, R14.

### Files touched

- `plans/INDEX.md`
- `plans/PR-NeMo-Requirements-Dedup.md`
- `plans/archive/PR-Resolution-Audit-S6A2-Authoritative-Source-Admission.md`
- `requirements.txt`
- `tests/test_nemo_requirements_ownership.py`

## Mechanism

Delete the four-line NeMo carve-out block from `requirements.txt`; do not edit
the canonical ASR direct reference. A focused contract test parses requirement
names rather than matching incidental whitespace, checks the exact ASR URL/SHA,
checks the standalone server's recovery instruction, and checks that the full
security sweep supplies both files to one `pip-audit` invocation. The test
proves both sides of ownership: a duplicate root declaration fails, and a
missing/moved ASR declaration fails. Pip dry-run supplies the executable
resolver proof. The merged #2071 plan move follows AGENTS.md teardown and does
not change runtime behavior.

## Intentional

- This slice removes the duplicate instead of pinning the PyPI line: matching a
  PyPI version to a direct git commit would still leave two authorities and may
  not describe the same code.
- General `requirements.txt` alone no longer installs the optional ASR model
  stack. That is intentional: ASR setup already requires
  `requirements.asr.txt`, and root Docker does not copy `asr_server.py`.
- The test treats Atlas Edge as an independent deploy surface with its own
  requirements; it does not centralize cross-product dependency files.
- No workflow edit is needed: `requirements.txt` already triggers the general
  consumers, and the scheduled full sweep already reads both files jointly.

## Deferred

- #2040B: generate and validate a full transitive constraints lock across the
  Python 3.10 and 3.11 fleets. The now-landed unit gate makes that follow-up
  testable, but it is a separate, much larger dependency-set decision.
- Issue #2040 remains open after this PR until the transitive-lock slice lands.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_nemo_requirements_ownership.py -q` before the
  requirements edit -- expected red: 1 failed, 6 passed because root still
  declared NeMo.
- `python -m pytest tests/test_nemo_requirements_ownership.py -q` after the
  requirements edit -- 7 passed.
- `python -m pytest tests/test_nemo_requirements_ownership.py
  tests/test_atlas_main_voice_startup.py
  tests/test_content_ops_ci_requirements_workflows.py -q` -- 38 passed, one
  pre-existing `pynvml` deprecation warning from Torch.
- `python -m pytest tests/test_check_unit_gate.py -q` -- 14 passed.
- Python 3.11 venv command: python -m pip install --dry-run --no-deps -r
  requirements.txt -r requirements.asr.txt -- pip resolved NeMo to commit
  `0f378e9d8dd72630c911025b555f18658d44cc8f` without a duplicate-source
  conflict; stopped during the unrelated Torch 532.2 MB wheel download after
  154.1 MB, and no package was installed.
- Command: git diff --exit-code origin/main -- requirements.asr.txt
  atlas_edge/requirements.txt .github/workflows/security_full_sweep.yml
  asr_server.py -- passed; canonical pin, independent Atlas Edge ownership,
  joint audit, and runtime recovery path are unchanged.
- `git diff --check` -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- `python scripts/sync_pr_plan.py plans/PR-NeMo-Requirements-Dedup.md
  origin/main --check` -- passed after generated metadata converged.
- Pending at push: guarded pre-push local review.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 3 |
| `plans/PR-NeMo-Requirements-Dedup.md` | 165 |
| `plans/archive/PR-Resolution-Audit-S6A2-Authoritative-Source-Admission.md` | 0 |
| `requirements.txt` | 4 |
| `tests/test_nemo_requirements_ownership.py` | 70 |
| **Total** | **242** |
