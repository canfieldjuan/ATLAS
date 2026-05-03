# Open Questions / Blockers

Last updated: 2026-05-03T20:03Z by claude-2026-05-03

Active questions that need an owner or a decision. Resolved questions move to [`decisions.md`](decisions.md) as new entries; never delete from this list without a corresponding decision. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

- **Future hardening (deferred)**: a CI check that requires any merged PR touching `extracted_*/` to also modify `COORDINATION.md` (or one of the `coordination/*.md` files). Forces the protocol mechanically instead of relying on convention. Land as a follow-up PR-Coord-2 once the doc has hit real friction (i.e. someone has demonstrably forgotten to update).
