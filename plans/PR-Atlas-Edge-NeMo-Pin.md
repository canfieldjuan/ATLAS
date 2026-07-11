# PR-Atlas-Edge-NeMo-Pin

## Why this slice exists

Issue #2040 remains open after #2072 and #2074 because Atlas Edge is an
independent deployment surface whose `nemo_toolkit[asr]` declaration still
floats. The independent review on #2072 explicitly called this out, the
operator carried it forward on #2040, and the merged #2074 plan names it as the
next dependency-hardening slice. An unchanged Edge checkout can therefore
select a different NeMo release and dependency contract over time.

Current-source reconstruction shows that a full Edge lock cannot yet be
truthful. `atlas_edge/pipeline/stt.py` executes NeMo Parakeet on CUDA and calls
the surface Jetson-oriented, while the repository README describes the Edge
node as Orange Pi or Jetson and names SenseVoice instead. There is no Edge
container, installer, accepted JetPack release, Python baseline, or vendor
Torch wheel in the repository. NVIDIA's Jetson installation contract requires
Torch to come from a JetPack-specific aarch64 wheel, so an ordinary PyPI lock
would silently replace that platform authority. This slice fixes the separable
NeMo-source drift without pretending that an x86 or generic-aarch64 Torch graph
is the deployed Jetson graph.

Resolver probes at the fixed #2074 package horizon show why silently taking the
last generic result is unsafe: conservative `manylinux_2_17` aarch64 backtracks
to NeMo 2.6.1, but pip-audit reports CVE-2026-24157 and CVE-2026-24159, both
fixed in 2.6.2. NeMo 2.6.2 raises the Torch floor to 2.6 and resolves for
`manylinux_2_28` aarch64 on both Python 3.10 and 3.11. NVIDIA's current Jetson
matrix provides Torch 2.7+ for JetPack 6.2, satisfying that floor. Pinning 2.6.2
therefore fails closed on older unsupported graphs instead of reproducibly
installing a known-vulnerable fallback.

### Problem-derived contract

- Root cause: Atlas Edge owns a separate NeMo ASR dependency but declares only
  its distribution name, so the selected NeMo code and its minimum Torch/Python
  contract change with the package index. This change fixes the root of the
  direct NeMo-source drift; it does not claim to fix the broader transitive
  graph, whose correct root is the missing accepted hardware/JetPack/Torch
  install contract.
- Review-finding root cause: the original plan treated the requirements file as
  the complete install contract but missed the runtime's ImportError remediation,
  which independently instructed operators to install the floating distribution.
  A pinned owner is not closed while its failure path advertises a bypass.
- Correct fix must touch/change: replace only the Edge NeMo declaration with the
  first release that fixes the known NeMo CVEs and that the newer aarch64 Python
  3.10/3.11 resolver accepts without choosing a vendor Torch wheel; extend the existing NeMo ownership contract
  test so unpinned, moved-version, duplicate, cross-surface authority, or
  floating runtime guidance fails; point the missing-dependency remediation at
  the pinned Edge requirements file;
  execute real aarch64 resolver probes; archive this session's merged #2074
  plan and refresh the plan index.
- Must not change: any other Edge dependency name/order/specifier; Torch source
  or version; root/standalone-ASR requirements or constraints; Edge inference
  behavior, Parakeet model/defaults, hardware claims, or product behavior; package indexes;
  Resolution Audit or Content Ops lanes; buyer-visible product shape; or the
  existing dirty files in the protected main checkout.

## Scope (this PR)

Ownership lane: dependency-hardening/atlas-edge-python
Slice phase: Production hardening

1. Pin the independent Atlas Edge NeMo ASR owner to `nemo_toolkit[asr]==2.6.2`
   while preserving every other requirement byte-for-byte.
2. Extend the existing NeMo requirements contract with positive and negative
   Edge pin cases, and prove the real requirement set resolves on conservative
   Linux aarch64 for Python 3.10 and 3.11.
3. Make the live missing-NeMo remediation install the pinned Edge requirements
   surface rather than advertising a floating package command.
4. Archive only this session's merged
   `PR-Python-Transitive-Constraints-Lock.md` and rebuild `plans/INDEX.md`.

Max files: 6

### Review Contract

- Acceptance criteria:
  - [ ] `atlas_edge/requirements.txt` contains exactly one NeMo distribution
        declaration and it is exactly `nemo_toolkit[asr]==2.6.2`.
  - [ ] Every non-NeMo line in the Edge requirements file keeps its current
        text, order, and specifier.
  - [ ] The existing root/standalone-ASR NeMo ownership contract and canonical
        git SHA remain unchanged.
  - [ ] Contract negatives reject an unpinned Edge declaration, a different
        version, a second Edge declaration, and reuse of the root/ASR git source.
  - [ ] The Edge STT missing-dependency message points operators at
        `pip install -r atlas_edge/requirements.txt` and no live Edge source
        advertises a floating NeMo install.
  - [ ] uv 0.10.10 resolves the real Edge requirement set with the fixed
        `2026-07-11T07:55:25Z` upload horizon for Python 3.10 and 3.11 on
        `aarch64-manylinux_2_28`, selecting NeMo 2.6.2 and Torch 2.13.0 on both
        generic-index targets without writing either result into the repo.
  - [ ] pip-audit no longer reports the two NeMo CVEs fixed by 2.6.2; unrelated
        pre-existing Transformer advisories are reported rather than widened
        into this direct-pin slice.
  - [ ] The focused NeMo contract and existing Atlas Edge unit tests remain
        green under the repository Python 3.11 test environment.
  - [ ] Only this session's merged #2074 plan is archived.
- Reachability proof: invoke uv against the real `atlas_edge/requirements.txt`
  for both supported probe interpreters and the conservative aarch64 platform;
  the observable result is a complete resolution selecting exactly NeMo 2.6.2.
  No new runtime surface is introduced, so hardware inference remains deferred.
- Affected surfaces: Atlas Edge Python dependency installation, its scheduled
  SCA input, repository NeMo ownership contracts, and plan teardown.
- Risk areas: selecting a NeMo release that raises the Torch floor, accidentally
  claiming authority over NVIDIA's JetPack Torch wheel, moving the root/ASR
  source, or describing a generic resolver probe as a hardware runtime proof.
- Reviewer rules triggered: R1, R2, R11, R12, R14.

### Files touched

- `atlas_edge/pipeline/stt.py`
- `atlas_edge/requirements.txt`
- `plans/INDEX.md`
- `plans/PR-Atlas-Edge-NeMo-Pin.md`
- `plans/archive/PR-Python-Transitive-Constraints-Lock.md`
- `tests/test_nemo_requirements_ownership.py`

## Mechanism

Change the single Edge NeMo line from a floating extra to an exact PyPI release.
The selected release is constrained by executable platform and security probes
rather than shared-source convenience: 2.6.2 is the first release fixing the
observed NeMo advisories and resolves on `manylinux_2_28` aarch64 for both
Python targets. The repository still does not select a specific JetPack Torch
wheel; NeMo's `torch>=2.6` requirement supplies the fail-closed compatibility
floor.

Extend `tests/test_nemo_requirements_ownership.py` rather than creating a second
requirements parser. The existing normalized distribution-name helper reads the
real Edge file, asserts one exact Edge owner, and parameterized negative cases
prove the contract rejects the drift shapes. A focused real-source assertion
also binds the ImportError remediation to that requirements authority and
rejects the old floating command. The root/ASR assertion remains a separate
authority because its git SHA describes a different deployment.

Local uv probes provide the executable dependency-boundary proof without
installing foreign-architecture wheels. The merged #2074 plan move follows the
worktree-first teardown contract and has no runtime effect.

## Intentional

- This is a direct NeMo pin, not a full transitive Edge lock. A lock that pins
  PyPI Torch would be actively misleading for Jetson because NVIDIA publishes
  JetPack-specific Torch wheels; the repository has not selected the hardware,
  JetPack, Python, or vendor-wheel baseline needed to compile the real graph.
- NeMo 2.6.2 is selected instead of the root/ASR git SHA. The shared SHA resolves
  as NeMo 3.1.0 and raises the aarch64 Torch floor, so copying it would create
  visual consistency without preserving the independent Edge Python contract.
- NeMo 2.6.2 is selected instead of the vulnerable 2.6.1 fallback and instead
  of newer NeMo lines that drop the current Python 3.10/3.11 compatibility.
  Older `manylinux_2_17`/Torch graphs now fail dependency resolution rather than
  silently backtracking across known CVEs.
- The runtime and README mismatch is evidence that a platform decision is
  missing, not permission to change Edge product/runtime shape in a dependency
  PR. This slice leaves inference/model behavior untouched and changes only the
  operator guidance required to preserve the dependency contract.

## Deferred

- #2040C2: after the operator accepts one real Edge deployment baseline
  (Orange Pi/SenseVoice or a named Jetson + JetPack + Python + vendor Torch
  source), generate and enforce that platform's complete transitive lock and
  run an import/model smoke on the target hardware or matching container.
- Other unpinned Edge direct dependencies remain part of that platform lock;
  this slice changes only the explicitly reviewed NeMo gap.
- The ten resolver-selected Transformer 4.53.3 advisories are recorded on
  issue #2040 at `issuecomment-4944339963`; their compatible fix belongs with
  the accepted Edge platform graph rather than an unrelated direct upgrade.

Parked hardening: none.

## Verification

- Pre-plan probe: uv 0.10.10 resolved the current unpinned Edge graph at the
  fixed upload horizon to NeMo 2.6.1 on Python 3.10 and 3.11 for
  `aarch64-manylinux_2_17` (184 and 180 package cells respectively).
- Security probe: pip-audit reported CVE-2026-24157 and CVE-2026-24159 in
  NeMo 2.6.1, both fixed in 2.6.2; this invalidated the first compatibility-only
  plan before push.
- Secure-platform probe: NeMo 2.6.2 rejects `manylinux_2_17` because it requires
  Torch 2.6+, then resolves on `aarch64-manylinux_2_28` for both Python 3.10 and
  3.11 (202 and 196 package cells, respectively).
- Pre-plan rejection probe: substituting the root/ASR git SHA failed resolution
  because it identifies NeMo 3.1.0, which requires `torch>=2.6` and has no
  matching conservative-aarch64 wheel in the generic index.
- Review reproduction: `rg` found the sole live bypass at
  `atlas_edge/pipeline/stt.py:93`, which printed floating
  `pip install nemo_toolkit[asr]`; reconciliation and `claude-review` correctly
  failed on that incomplete ownership boundary.
- Test-first focused contract: `python -m pytest
  tests/test_nemo_requirements_ownership.py -q` failed 1 / passed 11 against the
  floating Edge line, then passed 12 after the secure pin.
- Review-fix test first: the same focused file failed 1 / passed 12 while the
  runtime still advertised the floating command, then passed 13 after the
  remediation used the pinned Edge requirements file.
- `python -m pytest tests/atlas_edge -q` -- 144 passed.
- `python -m py_compile atlas_edge/pipeline/stt.py
  tests/test_nemo_requirements_ownership.py` -- passed; the multiline class
  sweep found no live floating NeMo install command under `atlas_edge`.
- uv 0.10.10 against the real pinned file and fixed upload horizon on
  `aarch64-manylinux_2_28` -- Python 3.10 resolved 202 cells and Python 3.11
  resolved 196; both selected NeMo 2.6.2 and Torch 2.13.0 from the generic
  index without installing foreign-architecture wheels.
- `uvx pip-audit -r atlas_edge/requirements.txt --progress-spinner off` -- no
  NeMo advisory remains; command exits 1 on the ten pre-existing Transformer
  4.53.3 advisories recorded on #2040.
- Exact non-NeMo Edge requirements comparison and forbidden root/ASR diff --
  byte-identical / no diff; Edge STT changed only the ImportError guidance.
- `scripts/check_ascii_python.sh` via Bash -- passed.
- `git diff --check` and `python scripts/sync_pr_plan.py
  plans/PR-Atlas-Edge-NeMo-Pin.md origin/main --check` -- passed after plan
  metadata converged.
- Pending at push: guarded pre-push local review.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_edge/pipeline/stt.py` | 3 |
| `atlas_edge/requirements.txt` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Atlas-Edge-NeMo-Pin.md` | 226 |
| `plans/archive/PR-Python-Transitive-Constraints-Lock.md` | 0 |
| `tests/test_nemo_requirements_ownership.py` | 37 |
| **Total** | **271** |
