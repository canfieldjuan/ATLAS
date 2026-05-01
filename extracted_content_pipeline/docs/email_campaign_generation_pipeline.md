# Email / Campaign Generation Pipeline

This maps the sellable campaign system currently living in `atlas_brain`.
The important distinction: copied Atlas files are source material, not the final
customer module. The product version must replace Atlas runtime imports with
product-owned ports and adapters before it can be shipped outside the repo.

## Product Shape

This is a data-backed outbound platform, closer to a Smartlead/Lemlist/Outreach
competitor than a single generation task. It combines intelligence selection,
reasoning-aware copy generation, quality checks, manual review, send
orchestration, suppression, webhook ingestion, follow-up sequencing, analytics,
and multi-vertical campaign support.

## Runtime Flow

1. Opportunity sourcing
   - B2B opportunities are assembled from vendor intelligence, account signals,
     review candidates, target lists, blog matches, and calibrated score
     components.
   - Amazon seller opportunities are assembled from product review/category
     intelligence and seller targets.

2. Blueprint and prompt assembly
   - `b2b_campaign_generation.py` builds channel/persona-specific campaign
     payloads.
   - `amazon_seller_campaign_generation.py` builds seller/category outreach
     payloads.
   - `_campaign_sequence_context.py` compacts prior sequence, signal, quote, and
     selling-context data for follow-up prompts.

3. LLM generation
   - Prompt contracts live under `skills/digest/`.
   - B2B generation uses `b2b_campaign_generation`, `b2b_vendor_outreach`, and
     `b2b_challenger_outreach`.
   - Sequence progression uses `b2b_campaign_sequence`,
     `b2b_vendor_sequence`, `b2b_challenger_sequence`,
     `b2b_onboarding_sequence`, and `amazon_seller_campaign_sequence`.

4. Quality gates
   - `campaign_quality.py` revalidates drafts for witness-backed specificity
     before approval, queueing, and send.
   - `campaign_specificity_backfill.py` repairs stored campaign metadata when
     policy changes.

5. Review and approval
   - `api/b2b_campaigns.py` exposes review queues, quality diagnostics,
     bulk approval/rejection, sequence details, suppression management,
     generation, export, and analytics.
   - `api/seller_campaigns.py` exposes seller target and seller campaign
     endpoints.

6. Send orchestration
   - `campaign_send.py` sends queued campaigns through the configured sender
     after suppression, fatigue, send-window, and quality checks.
   - `campaign_sender.py` abstracts Resend and SES.
   - `campaign_suppression.py` manages email/domain suppression.

7. Engagement and progression
   - `campaign_webhooks.py` ingests ESP events, updates campaign/sequence state,
     records opens/clicks/bounces/unsubscribes/complaints, and adds suppressions.
   - `campaign_sequence_progression.py` generates next-step follow-ups from
     prior messages and engagement context.

8. Analytics and audit
   - `campaign_audit.py` writes immutable state-change events.
   - `campaign_analytics_refresh.py` refreshes `campaign_funnel_stats`.
   - Outcome, score component, timing, and audit metadata migrations preserve
     funnel attribution and later scoring loops.

## Source Inventory

Already staged in the current extraction scaffold:

- `autonomous/tasks/b2b_campaign_generation.py`
- `autonomous/tasks/campaign_audit.py`
- `autonomous/tasks/_campaign_sequence_context.py`

Still to extract after ports/adapters exist:

Core generation and orchestration:

- `autonomous/tasks/amazon_seller_campaign_generation.py`
- `autonomous/tasks/campaign_send.py`
- `autonomous/tasks/campaign_sequence_progression.py`
- `autonomous/tasks/campaign_suppression.py`
- `autonomous/tasks/campaign_analytics_refresh.py`

Services:

- `services/campaign_quality.py`
- `services/campaign_reasoning_context.py`
- `services/campaign_sender.py`
- `services/campaign_specificity_backfill.py`

API surfaces:

- `api/b2b_campaigns.py`
- `api/seller_campaigns.py`
- `api/campaign_webhooks.py`

Prompt contracts:

- `skills/digest/b2b_campaign_generation.md`
- `skills/digest/b2b_vendor_outreach.md`
- `skills/digest/b2b_challenger_outreach.md`
- `skills/digest/b2b_campaign_sequence.md`
- `skills/digest/b2b_vendor_sequence.md`
- `skills/digest/b2b_challenger_sequence.md`
- `skills/digest/b2b_onboarding_sequence.md`
- `skills/digest/amazon_seller_campaign_generation.md`
- `skills/digest/amazon_seller_campaign_sequence.md`

Campaign schema copied into this scaffold:

- `storage/migrations/066_b2b_campaigns.sql`
- `storage/migrations/068_campaign_sequences.sql`
- `storage/migrations/069_campaign_analytics.sql`
- `storage/migrations/070_campaign_suppressions.sql`
- `storage/migrations/073_campaign_sequence_fixes.sql`
- `storage/migrations/074_campaign_target_modes.sql`
- `storage/migrations/075_amazon_seller_campaigns.sql`
- `storage/migrations/090_audit_log_metadata_index.sql`
- `storage/migrations/104_campaign_outcomes.sql`
- `storage/migrations/146_campaign_score_components.sql`
- `storage/migrations/150_campaign_engagement_timing.sql`

Campaign-adjacent migrations intentionally deferred from the core scaffold:

- `storage/migrations/080_b2b_alert_baselines.sql` (also depends on tenant
  alert tables and `saas_accounts`)
- `storage/migrations/106_score_calibration.sql` (model calibration slice)
- `storage/migrations/235_vendor_targets_account_scope.sql`
- `storage/migrations/255_anthropic_message_batches.sql` (batch LLM replay
  infrastructure)

## External Providers And Env

- Current Atlas source uses Atlas LLM registry and pipeline helpers.
- Product target needs a standalone LLM client port with OpenAI/Anthropic
  adapters.
- Optional Anthropic Message Batches for campaign generation replay and
  reconciliation.
- ESP sending through Resend or AWS SES.
- Resend/Svix webhook signing for inbound events.
- Campaign sequence config currently under `ATLAS_CAMPAIGN_SEQ_*`.
- B2B campaign config currently under `ATLAS_B2B_CAMPAIGN_*`.
- Amazon seller campaign config currently under `ATLAS_SELLER_CAMPAIGN_*`.
- Product target should rename config away from Atlas-prefixed env vars before
  customer deployment.

## Extraction Debt

This scaffold still leans on Atlas framework modules. The biggest dependency
clusters are auth/tenant scope, scheduled-task models, DB pool lifecycle,
visibility events, LLM routing, skills registry, B2B intelligence helpers,
vendor/account opportunity selection, and Anthropic exact-cache/batch helpers.

The next extraction step should turn this map into seams:

1. Define product-local interfaces for DB access, LLM, sender, auth scope, and
   audit events.
2. Replace direct Atlas imports in copied modules with those interfaces.
3. Add import smoke coverage for campaign modules once the interfaces exist.
4. Add fixture-level tests around review queue, suppression, send, webhook,
   sequence progression, and analytics refresh.

## Product-Owned Modules Started

- `campaign_ports.py`: host-facing interfaces and shared dataclasses.
- `campaign_suppression.py`: standalone suppression policy using
  `SuppressionRepository`.
- `campaign_sequence_context.py`: standalone sequence context compaction using
  explicit `SequenceContextLimits`.
- `campaign_sender.py`: standalone Resend/SES sender adapters using explicit
  provider config.
- `campaign_send.py`: standalone due-send orchestrator using repository,
  suppression, sender, audit, and clock ports.
- `campaign_webhooks.py`: standalone Resend/Svix signature verification,
  webhook event normalization, event recording, and webhook-driven suppression
  policy using product ports.
- `campaign_analytics.py`: standalone analytics refresh orchestration using
  repository, audit, and visibility ports.
- `campaign_sequence_progression.py`: standalone due-sequence follow-up
  generation using sequence repository, LLM, skill-store, context, and audit
  ports.
- `campaign_generation.py`: standalone draft generation shell using
  intelligence, campaign repository, LLM, and skill-store ports.
