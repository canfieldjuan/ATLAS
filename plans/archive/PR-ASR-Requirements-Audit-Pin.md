# PR-ASR-Requirements-Audit-Pin

## Why this slice exists

The security guardrail adoption parked one Python SCA gap:
`requirements.asr.txt` was excluded from the advisory pip-audit matrix because
it installed `nemo_toolkit[asr]` from `NVIDIA/NeMo@main`. A moving VCS ref means
the dependency graph can change on every scheduled run without an Atlas commit,
so the safest first adoption choice was to exclude the file. That leaves the
ASR stack with no pip-audit coverage.

Root cause: the ASR requirements file points at a mutable upstream branch and
the security workflow intentionally audits only deterministic requirement
inputs. This fixes the root by pinning the NeMo VCS requirement to the concrete
commit currently behind upstream `main`, then adding `requirements.asr.txt`
back to the advisory pip-audit matrix.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Replace the mutable `NVIDIA/NeMo@main` ASR requirement with the resolved
   commit `0f378e9d8dd72630c911025b555f18658d44cc8f`.
2. Add `requirements.asr.txt` to the scheduled/main `pip-audit` matrix now
   that the file is deterministic.
3. Update security hardening docs so the ASR audit gap is no longer listed as
   deferred.

### Review Contract

Acceptance criteria:

- `requirements.asr.txt` no longer references `@main`.
- The NeMo requirement remains a VCS install from `NVIDIA/NeMo`, but uses
  commit `0f378e9d8dd72630c911025b555f18658d44cc8f`.
- `.github/workflows/security_guardrails.yml` includes `requirements.asr.txt`
  in the `pip-audit` requirements matrix.
- `docs/SECURITY_GUARDRAILS.md` and `HARDENING.md` no longer claim that
  `requirements.asr.txt` is parked or has zero CVE coverage.

Affected surfaces:

- Advisory scheduled/main Python SCA coverage.
- ASR dependency reproducibility.
- Security guardrail documentation and hardening queue.

Risk areas:

- Pinning the current commit preserves today's upstream code but stops silent
  branch movement; future NeMo updates need an explicit PR.
- `pip-audit` remains advisory/continue-on-error in this workflow, so this
  broadens visibility without making main red on the existing backlog.

Triggered reviewer rules:

- R1 Requirements match
- R2 Test evidence
- R3 Security/auth
- R8 CI/workflow safety
- R14 Codebase verification

### Files touched

- `.github/workflows/security_guardrails.yml`
- `HARDENING.md`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-ASR-Requirements-Audit-Pin.md`
- `plans/archive/PR-Workflow-Service-Image-Digests.md`
- `requirements.asr.txt`

## Mechanism

The NeMo requirement keeps the same package, extras, repository, and PEP 508
direct-reference shape, but swaps the mutable branch name for the commit that
GitHub reported for `refs/heads/main` on 2026-06-17:

```text
git ls-remote https://github.com/NVIDIA/NeMo.git refs/heads/main
0f378e9d8dd72630c911025b555f18658d44cc8f refs/heads/main
```

The scheduled/main `pip-audit` job already iterates a requirements-file matrix
with `continue-on-error: true`. Adding `requirements.asr.txt` there gives the
ASR stack advisory coverage without changing pull-request latency, because the
SCA job is skipped for `pull_request` and `pull_request_target` events.

## Intentional

- This pins to the current upstream commit rather than choosing a NeMo release
  tag. A tag migration may be the better long-term dependency policy, but a
  commit pin is the lowest-risk way to stop moving-input drift without changing
  ASR code behavior in the same slice.
- This does not ratchet pip-audit to blocking. The guardrail remains
  advisory/report-only until the known backlog is triaged.
- This does not touch `requirements.txt`'s PyPI `nemo_toolkit[asr]` line;
  this slice is scoped to the parked ASR audit-input gap.

## Deferred

- Future NeMo updates should come through an explicit dependency update PR that
  changes the pinned commit or migrates to a release tag after ASR smoke
  verification.

Parked hardening: none; this drains the "Pin or retire floating ASR dependency
audit input" item from `HARDENING.md`.

## Verification

- `git ls-remote https://github.com/NVIDIA/NeMo.git refs/heads/main` --
  passed; `main` resolved to
  `0f378e9d8dd72630c911025b555f18658d44cc8f`.
- `rg -n 'NVIDIA/NeMo\.git@(main|0f378e9d8dd72630c911025b555f18658d44cc8f)|requirements\.asr\.txt|floating NVIDIA/NeMo@main|zero CVE' requirements.asr.txt .github/workflows/security_guardrails.yml docs/SECURITY_GUARDRAILS.md HARDENING.md`
  -- passed; only the pinned NeMo commit and expected `requirements.asr.txt`
  workflow/docs references remain.
- Requirements parser smoke for `requirements.asr.txt` -- passed; pip's
  requirement parser accepts the direct reference and no parsed requirement
  contains `@main`.
- YAML parse smoke for `.github/workflows/security_guardrails.yml` -- passed.
- `python scripts/sync_pr_plan.py --check plans/PR-ASR-Requirements-Audit-Pin.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/security_guardrails.yml` | 1 |
| `HARDENING.md` | 9 |
| `docs/SECURITY_GUARDRAILS.md` | 9 |
| `plans/INDEX.md` | 3 |
| `plans/PR-ASR-Requirements-Audit-Pin.md` | 135 |
| `plans/archive/PR-Workflow-Service-Image-Digests.md` | 0 |
| `requirements.asr.txt` | 2 |
| **Total** | **159** |
