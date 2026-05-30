# PR-FAQ-Deflection-Blob-DNS-Pinning

## Why this slice exists

#1169 closed blob-submit redirect SSRF, but its review called out a remaining
DNS-rebinding time-of-check/time-of-use gap: the submit endpoint validated the
hostname's resolved IPs before fetch, then let `urllib` resolve the hostname
again when opening the socket. `HARDENING.md` parked this as the active
customer-reachable security item for the deflection/Stripe lane.

This slice promotes that parked hardening item because the portfolio funnel now
passes signed blob URLs into ATLAS and the endpoint should fail closed before
broader hosted validation.

The total diff exceeds the 400 LOC target because the DNS-pinning socket path,
the promoted hardening removal, and the security regression fixtures need to
ship together; splitting the tests from the transport change would leave the
SSRF guard's failure branch unproven.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-blob-fetch
Slice phase: Production hardening

1. Pin the HTTPS blob fetch socket to an IP returned by the preflight DNS
   validation.
2. Preserve TLS verification and SNI against the original blob hostname.
3. Keep the existing no-redirect, bounded-read, HTTPS-only, no-credential, and
   private-address rejection behavior.
4. Remove the promoted DNS-rebinding entry from `HARDENING.md`.

### Files touched

- `plans/PR-FAQ-Deflection-Blob-DNS-Pinning.md`
- `HARDENING.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The HTTPS blob target validator now returns a structured blob-fetch target
that includes the normalized URL, parsed hostname/port, request path, Host
header, and a connect host selected from the already-validated public DNS
answers. The opener uses a pinned HTTPS connection to that connect host while
sending SNI and certificate verification for the original
hostname and preserving the original Host header.

Redirects remain fail-closed because the code reads the first response directly
and rejects any 3xx status without following `Location`.

## Intentional

- This slice does not add a trusted-host allowlist. The immediate parked issue
  is the DNS-rebinding TOCTOU gap; a host allowlist would be a separate product
  contract with the portfolio uploader.
- The resolver still rejects the whole target when any DNS answer is private or
  otherwise blocked, even if another answer is public.
- The implementation uses stdlib `http.client`/`ssl` instead of adding a new
  HTTP dependency to the extracted package.

## Deferred

- Parked hardening: none for this slice. The DNS-rebinding item from #1169 is
  promoted and removed from `HARDENING.md`.
- Portfolio result-page hosted validation remains in the deflection/Stripe
  lane after the open hosted-submit handoff thread finishes.

## Verification

- python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py - passed.
- python -m pytest tests/test_extracted_content_deflection_submit.py -q - 12 passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- bash scripts/run_extracted_pipeline_checks.sh - passed, 2812 passed, 10 skipped, 1 warning.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-deflection-blob-dns-pinning.md - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 75 |
| Blob fetch implementation | 164 |
| Regression tests | 198 |
| Hardening queue update | 10 |
| **Total** | **447** |

The non-plan code/test diff is under 400 LOC. The total exceeds 400 because
the plan doc plus security regression fixtures ship with the code change.
