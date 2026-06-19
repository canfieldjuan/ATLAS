# Backstop Green Follow-Up

## Scope

Keep #1712 focused on the advisory unit backstop/auditor workflow while using follow-up slices to make that backstop green.

## Current slice

This slice hardens pytest collection so the unit backstop uses the real `asyncpg` module when dependencies are installed and excludes known live/service-backed DB tests through the existing `integration` marker boundary.

## Deferred

Residual unit failures should be triaged after the backstop is rerun with this harness fix in place. Security Guardrails `startup_failure` remains workflow hardening outside this cleanup slice.
