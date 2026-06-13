Juan Canfield
Effingham, IL
(630) 750-3509 | canfieldjuan24@gmail.com
LinkedIn: https://www.linkedin.com/in/juan-canfield-9b2a733b5/
GitHub: https://github.com/canfieldjuan/atlas-portfolio
Authorized to work in the U.S.

---

FOUNDING / STAFF AI PLATFORM ENGINEER

Builder of AI products *and* the platform substrate underneath them. I ship
end-to-end systems where the engineering decisions — deterministic vs. LLM,
where the paywall lives, how spend is bounded — are the product. 3+ years
turning LLMs into reliable, auditable operators inside real revenue and
content workflows, across Python, TypeScript/React, and PostgreSQL.

---

SELECTED PROJECTS

AI Content Ops Platform  —  multi-asset generation engine (~77K LOC, standalone package)
A tenant-scoped engine that generates blog posts, B2B campaigns, landing pages,
reports, sales briefs, and FAQ deflection reports from structured customer data.
- Extracted the entire generation subsystem into a standalone package behind
  typed ports (LLM client / skill store / intelligence repository) so generators
  build and test independently of the 50+-router host — and the LLM, data, and
  skill backends swap without forking a single generator.
- Enforced extraction discipline with a manifest-driven byte-level sync audit
  and hard import guards, so the package can't silently drift from or hard-depend
  on the host monolith.
- Gated every asset behind a deterministic quality catalogue plus a human review
  API, so unsafe or low-quality output is blocked before publish, not caught
  after — and surfaced per-tenant usage/budget so generation spend is attributable
  per customer.

Deflection Report  —  self-serve, paid AI product (upload -> free snapshot -> Stripe -> full report)
A monetized vertical on the Content Ops platform: customers upload a support-ticket
export, get a free FAQ "snapshot" teaser, and pay to unlock the full report.
- Built the core pipeline deterministic and LLM-free end to end (clustering and
  answer drafting), so customer ticket PII never leaves the stack and the product
  carries zero marginal token cost — making the "100% deterministic" privacy claim
  literally true and the report sellable to privacy-sensitive buyers.
- Enforced the paywall at the data-access layer (keyed on account+request), with
  the free snapshot as a server-side projection that withholds the full artifact —
  so the gate can't be bypassed client-side and the teaser leaks no paid content.
- Made the money path safe under failure: signature-verified, idempotent Stripe
  webhooks (event de-dup) with a single-statement atomic paid transition, and a
  delivery worker that claims jobs with row-level locks + stale reclaim — no
  double-charge, no double-send, crash-safe at-least-once report delivery.

LLM Cost & Caching Infrastructure  —  spend made observable, attributable, and bounded
The platform substrate under every LLM call: 8 providers behind one router with
hybrid local/cloud fallback, three caching tiers, and full cost closure.
- Designed a 3-tier cache — exact (Postgres), semantic (evidence-hash with
  exponential confidence decay so hits go stale gracefully), and a *declared*
  cache-strategy registry that makes coverage a deliberate design decision —
  cutting repeat-token cost ~70% on warmed runs without silent staleness.
- Closed the "cache hits live only in memory" gap with a savings ledger that
  records the counterfactual cost of every hit and rolls it up per tenant, turning
  cache wins into auditable dollars instead of a guess.
- Built the differentiated reliability layer few teams have: per-day cost-drift
  reconciliation of locally-traced spend against each provider's invoiced billing
  API, plus a runtime budget gate with per-tenant caps that throttles a runaway
  customer without affecting anyone else, plus Anthropic message-batching for
  half-cost on batchable workloads.

---

EXPERIENCE

AI/ML Solutions Architect — FineTune Lab.AI | Aug 2023 - Present
- Shipped production AI workflows integrating multiple LLM providers, custom
  inference (vLLM / Hugging Face / Ollama), and full-stack product systems.
- Built the observability and cost-closure layer above — tracing, drift
  reconciliation, budget gating, and caching — so AI spend and failures became
  visible and bounded in production.
- Designed PostgreSQL data architecture (state models, Row-Level Security, query
  tuning) and operator review systems so AI outputs could be audited and corrected.

ATLAS / Churn Signals — AI churn-intelligence & reasoning platform (project)
- Built a synthesis-first agentic reasoning pipeline over 40,000+ raw / 30,000+
  enriched reviews across 56 vendors, exposed as an 80+-tool integration surface
  feeding battle cards, challenger briefs, and tenant churn reports.
- Added retry handling, validation-result persistence, and operator correction so
  generated intelligence could be trusted, audited, and reused in live workflows.

---

ENGINEERING PRACTICE
Ship non-trivial work as thin, planned, single-purpose changes reviewed by an
independent agent against a mechanical gate chain (rules contract, AI-finding
reconciliation, automated rule-trigger enforcement) — every escaped defect
converted into a durable, test-backed gate.

---

SKILLS
LLMs, Agentic Workflows, LLM Cost/Caching & Observability, Multi-Provider Routing
(OpenAI, Claude, OpenRouter, Groq, Together, vLLM, Ollama), RAG & Knowledge Graphs,
MCP Servers, Prompt/Skill Engineering, Quality Gating, Stripe/Payments,
Python, TypeScript, React, Node.js, SQL/PostgreSQL, Redis, Docker, CI/CD, System Design

EDUCATION
Associate in Computer Aided Drafting and Design — ITT Technical Institute, Orland Park, IL
