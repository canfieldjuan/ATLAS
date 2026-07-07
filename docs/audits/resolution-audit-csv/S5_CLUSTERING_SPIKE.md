# S5 Clustering Current-Code Spike

Issue: https://github.com/canfieldjuan/ATLAS/issues/1993

This is a current-code calibration note for S5. It does not propose a new
report shape and does not change the buyer-visible Resolution Audit output.
The goal is to pin the behavior that the implementation slice must correct.

## Root Cause

The support-ticket pipeline promotes deterministic token-set labels into the
FAQ topic partition before question-level similarity can evaluate the full
ticket set. The downstream question subcluster and optional embedding booster
operate only inside that hard topic partition.

## Current-Code Observations

| Probe | Current result | Why it matters |
|---|---|---|
| 12 SSO/SAML login tickets with one shared intent | 10 rows become `login`, two rows become singleton token-set topics, and the FAQ renders zero repeat rows | Same-intent language can fall below the repeat gate before any buyer sees it |
| Same SSO/SAML fixture with a fake embedding port | The port receives only the 10-row `login` partition, never the two singleton topic partitions; zero rows render | Turning embeddings on is not enough because the hard partition still hides fragments |
| 10 cancel-subscription + 10 cancel-order tickets | The preview top cluster is `order` with 17 rows; the FAQ renders one mixed 11-ticket buyer row and excludes 9 rows | A shared surface token can merge distinct workflows before the report chooses a representative |
| Two repeated refund questions, input order reversed | Item order and source-id order reverse with input order | Representative/output ordering is not yet order-shuffle stable |
| Token-set rows above the clustering threshold | Rows remain included but uncategorized, with skip diagnostics | S5 implementation must either stay below the skip path for sync submits or keep the warning/load boundary visible |

## Current-Code Proof

`tests/test_resolution_audit_s5_clustering_spike.py` exercises the real
support-ticket package builder, real FAQ markdown builder, and real token-set
clusterer. The tests intentionally assert current behavior. The follow-up S5
implementation PR should replace those current-behavior assertions with the
accepted corrected behavior instead of adding another downstream patch.

## Implementation Constraints For The Next Slice

- Do not change report, snapshot, landing, email, or PDF shape without operator
  approval.
- Do not fix this by tuning a lexical threshold alone; the spike confirms the
  problem sits at the hard partition boundary.
- Do not ship a single-link semantic merge. The existing investigation artifact
  already shows the cancel fixture is single-link fragile.
- Keep the large-upload token-set skip visible, or prove the sync submit cap
  prevents this path from mattering for the synchronous audit flow.
