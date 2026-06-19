# PR-Graphiti-Python-3-14-Base-Image

## Why this slice exists

Dependabot proposed moving the Graphiti local deployment image from
`python:3.11-slim` to `python:3.14-slim`. This image installs Graphiti,
FastAPI, sentence-transformers, PyTorch CPU, and Neo4j/Postgres clients at build
time, so the base-image bump needs dependency-wheel evidence before it can be
treated as a safe one-line Dockerfile change.

## Scope (this PR)

Ownership lane: graphiti/runtime-image
Slice phase: Production hardening

1. Update `Dockerfile.graphiti` from `python:3.11-slim` to `python:3.14-slim`.
2. Verify the Graphiti Docker requirements can resolve CPython 3.14 Linux wheels,
   including PyTorch CPU and compiled dependencies.
3. Carry the shared Security Guardrails workflow repair already proven on the
   adjacent Dependabot branches.

### Files touched

- `Dockerfile.graphiti`
- `.github/workflows/security_guardrails.yml`
- `plans/PR-Graphiti-Python-3-14-Base-Image.md`

### Review Contract

Acceptance criteria:

- [ ] `Dockerfile.graphiti` uses `python:3.14-slim`.
- [ ] The Graphiti Docker dependency stack has CPython 3.14 Linux wheel coverage
      for the runtime-critical compiled dependencies: `torch`, `numpy`,
      `pydantic-core`, `asyncpg`, `scipy`, and `scikit-learn`.
- [ ] No Graphiti application code, runtime command, port, healthcheck, or
      requirements file changes are included in this slice.
- [ ] PR-level guardrails have the same workflow fix used by the recently green
      Dependabot branches.

Affected surfaces: Docker runtime image / local Graphiti deployment / CI guardrails.

Risk areas: dependency compatibility / runtime image build / deployment safety.

Reviewer rules triggered: R1, R2, R11, R12, R14.

## Mechanism

`Dockerfile.graphiti` keeps the existing build flow: install system `git`/`curl`,
install PyTorch CPU from the configured PyTorch CPU index, install the remaining
`graphiti-wrapper/requirements.txt` entries from PyPI, copy the wrapper, switch
to the non-root `atlas` user, and run `uvicorn`. Only the Python base image tag
changes.

The dependency-risk check is a wheel-resolution probe for the same requirements
under CPython 3.14 Linux tags. It confirms the current resolver can obtain
binary wheels for the compiled dependencies that would otherwise be likely build
breakers.

## Intentional

- Docker base-image maintenance only; no Graphiti service code or dependency
  pin changes.
- Keeps the existing `TORCH_CPU_SPEC` and `TORCH_CPU_INDEX_URL` build arguments.
- Does not add a Docker build workflow; this PR documents the local wheel probe
  and leaves full image-build CI as a follow-up if the Graphiti image becomes a
  release-blocking artifact.

## Deferred

- Add a dedicated Docker build smoke for `Dockerfile.graphiti` if/when the
  Graphiti local image becomes a CI-enforced release artifact.

## Parked hardening

None.

## Verification

- PyTorch CPU index probe confirmed releases are available for the resolver line.
- CPython 3.14 Linux wheel probe passed for the PyTorch CPU package.
- Mixed manylinux CPython 3.14 wheel-resolution probe passed for the Graphiti
  Docker requirements, including `torch`, `numpy`, `pydantic-core`, `asyncpg`,
  `scipy`, and `scikit-learn`.
- Docker build was not run in this environment; validation is by dependency
  wheel resolution plus CI.

## Estimated diff size

| File | LOC |
|---|---:|
| `Dockerfile.graphiti` | ~1 |
| `.github/workflows/security_guardrails.yml` | shared repair |
| `plans/PR-Graphiti-Python-3-14-Base-Image.md` | ~80 |
| **Total** | **~106** |
