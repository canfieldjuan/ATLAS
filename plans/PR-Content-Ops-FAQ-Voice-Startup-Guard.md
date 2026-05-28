# PR-Content-Ops-FAQ-Voice-Startup-Guard

## Why this slice exists

`HARDENING.md` records that local FAQ route validation can look broken on CUDA-less hosts because Atlas startup tries to auto-start the voice ASR subprocess with `asr_device="cuda"`. Non-voice Content Ops route validation should not wait on or fail because a local machine cannot spawn the optional CUDA ASR server.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Guard ASR subprocess auto-start when the configured ASR device requires CUDA but local CUDA is unavailable.
2. Preserve existing behavior when an external ASR server is already running.
3. Enroll the focused regression test in a dedicated CI workflow.
4. Remove the resolved `HARDENING.md` entry.

### Files touched

- `atlas_brain/main.py`
- `tests/test_atlas_main_voice_startup.py`
- `.github/workflows/atlas_main_voice_startup_checks.yml`
- `HARDENING.md`
- `plans/PR-Content-Ops-FAQ-Voice-Startup-Guard.md`

## Mechanism

`_start_asr_server` already checks whether the configured ASR endpoint is healthy before spawning `asr_server.py`. This slice keeps that order, then adds a device guard before `subprocess.Popen`:

```python
reason = _asr_autostart_blocked_reason(settings.voice.asr_device)
if reason:
    logger.warning("%s", reason)
    return None
```

The guard only applies to `cuda*` devices. CPU ASR and already-running external ASR endpoints remain unaffected.

## Intentional

- This does not change the default voice settings; it only prevents an impossible local CUDA spawn.
- This does not disable the voice pipeline globally. The issue being drained is the blocking ASR subprocess auto-start path.
- This does not solve the separate startup migration warning entry.

## Deferred

- Future PR: handle `b2b_campaigns.updated_at` startup migration drift.
- Parked hardening: none. This slice removes the voice ASR startup item from `HARDENING.md`.

## Verification

Passed locally:

```bash
python -m py_compile atlas_brain/main.py tests/test_atlas_main_voice_startup.py
python -m pytest tests/test_atlas_main_voice_startup.py -q
4 passed, 1 torch CUDA import deprecation warning from the local environment
```

To be run before PR:

```bash
bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-voice-startup-guard-pr-body.md
```

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 74 |
| ASR startup guard | 36 |
| Tests | 89 |
| CI workflow | 37 |
| Hardening cleanup | 9 |
| **Total** | **245** |
