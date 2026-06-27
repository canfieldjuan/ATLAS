# PR-ASR-Pip-Audit-Egg-Fragment

## Why this slice exists

`PR-ASR-Requirements-Audit-Pin` added `requirements.asr.txt` to the advisory
pip-audit matrix to close an ASR Python SCA gap, but the pinned NeMo line still
carries a trailing `#egg=nemo_toolkit[asr]` URL fragment. pip-audit's resolver
rejects that fragment (`invalid-egg-fragment`) and aborts before auditing the
file, so the ASR dependency tree (torch, torchaudio, nemo_toolkit and its
transitive ML stack) gets zero CVE coverage. On the scheduled/main
`security_guardrails.yml` run the failure is hidden by `continue-on-error`, so
the matrix entry looks green while auditing nothing; only the no-soft-fail
`security_full_sweep.yml` SCA job surfaces it as red.

Root cause: the requirement uses PEP 508 `name[extra] @ URL` direct-reference
form, which already declares the package and extras, AND a redundant legacy
`#egg=` fragment. pip tolerates the duplication; pip-audit does not. This fixes
the root by dropping the redundant fragment so the requirement is a clean direct
reference that pip-audit can resolve. The install result is byte-for-byte the
same package at the same commit, so this is behavior-neutral and unblocks the
audit that `PR-ASR-Requirements-Audit-Pin` intended to enable.

This is the prerequisite for ratchet slice R2, tracked in the Security Scanner
Ratchet plan (PR #1837; its plan doc lands on `main` when that PR merges). It
deliberately does not flip pip-audit from advisory to blocking; that waits
until the now-running audit's findings are measured and triaged.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Edit `requirements.asr.txt`: drop the trailing `#egg=nemo_toolkit[asr]`
   fragment from the NeMo direct reference, leaving the pinned commit intact.
2. Add this plan doc.

No workflow change is needed: `requirements.asr.txt` is already enrolled in both
the `.github/workflows/security_guardrails.yml` pip-audit matrix and the
`.github/workflows/security_full_sweep.yml` SCA job. No doc change is needed:
`docs/SECURITY_GUARDRAILS.md` already lists the file as audited, and this fix
makes that claim true rather than aspirational.

### Review Contract

Acceptance criteria:

- `requirements.asr.txt` installs the same NeMo commit
  (`0f378e9d8dd72630c911025b555f18658d44cc8f`) as before; only the redundant
  `#egg=` fragment is removed.
- The requirement line is valid PEP 508 and parses with
  `packaging.requirements.Requirement`.
- No workflow, doc, or product-code change ships in this PR.

Affected surfaces:

- Advisory Python SCA coverage for the ASR dependency tree (it begins to
  actually run instead of erroring out).
- ASR install reproducibility (unchanged).

Risk areas:

- Dropping `#egg=` could in principle change the resolved package name: it does
  not, because the `nemo_toolkit[asr] @` prefix already supplies name + extras
  (verified by parsing the line with `packaging.requirements.Requirement`).

Triggered reviewer rules:

- R1 Requirements match
- R8 CI/workflow safety
- R14 Codebase verification

### Files touched

- `plans/PR-ASR-Pip-Audit-Egg-Fragment.md`
- `requirements.asr.txt`

## Mechanism

Before: `nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@<sha>#egg=nemo_toolkit[asr]`.
After: the same line without the trailing `#egg=nemo_toolkit[asr]`.

pip parses both forms to the identical direct-URL requirement (the `name @ URL`
prefix is authoritative for a PEP 508 direct reference; the `#egg=` fragment is
legacy metadata pip only needs for bare `git+...#egg=name` specs). pip-audit,
however, runs the requirement set through pip's stricter resolver, which raises
`invalid-egg-fragment` on a `#egg=name[extra]` fragment and aborts the whole
file. Removing the fragment makes the audit proceed across the ASR tree.

The audit stays advisory in this slice. The scheduled/main matrix keeps
`continue-on-error`, and the nightly SCA job keeps failing red on real
findings if any exist -- which is now the point: the next run reveals whether
the ASR tree carries known-vulnerable dependencies. Flipping advisory to
blocking is deferred to ratchet slice R2.

## Intentional

- Minimal, behavior-neutral: one fragment removed, no version bump, no pin
  change, no workflow/doc edit.
- Advisory posture preserved: this unblocks measurement, it does not gate. The
  blocking flip and any dependency upgrades belong to R2 once findings are known.
- No `plans/INDEX.md` entry: in-flight plans live in the `plans/` root; INDEX is
  the archive index and is updated when the slice is archived after merge.

## Deferred

- Ratchet slice R2 (tracked in PR #1837): measure the ASR pip-audit findings,
  triage/upgrade/pin or add justified ignores, then drop `continue-on-error` to
  make the ASR SCA blocking.

Parked hardening: none.

## Verification

- PEP 508 parse proof (local): load the seventh line of `requirements.asr.txt`
  through `packaging.requirements.Requirement` -- it parses with name
  `nemo_toolkit`, extra `asr`, and no egg fragment.
- Format gates (local): `scripts/sync_pr_plan.py --check` on this plan doc,
  `scripts/check_ascii_python.sh`, and `git diff --check` all pass.
- End-to-end (CI, authoritative): the `.github/workflows/security_full_sweep.yml`
  SCA job and the `.github/workflows/security_guardrails.yml` pip-audit
  `requirements.asr.txt` matrix entry stop erroring with `invalid-egg-fragment`
  and instead either pass or surface real
  ASR-tree advisories. Full pip-audit is not run locally because it resolves
  NeMo from a git commit (heavy / network-bound); CI owns that proof.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-ASR-Pip-Audit-Egg-Fragment.md` | 132 |
| `requirements.asr.txt` | 2 |
| **Total** | **134** |
