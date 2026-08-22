-- atlas: atomic-bookkeeping
-- Durable, tenant-scoped missed-call recovery for EOM residential estimate leads.
--
-- A public form acknowledgement is a separate transactional message. These
-- tables begin only after an authenticated office operator records a real
-- unanswered call. The current sequence is mutable state; every meaningful
-- transition is preserved in the append-only event ledger.
--
-- Rollback: stop the worker and the new routes first, then retain this
-- additive evidence. Never delete attempts, sequence events, or sent-step
-- evidence during an ordinary application rollback.

-- All operator mutations bind their Idempotency-Key globally before any
-- contact-specific work. A stale browser retry whose route changes from lead
-- A to lead B must fail closed rather than recording a second call or queueing
-- customer mail for the wrong lead.
CREATE TABLE IF NOT EXISTS eom_missed_call_operation_receipts (
    operation_key VARCHAR(128) PRIMARY KEY,
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    operation_kind VARCHAR(32) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_missed_call_operation_receipts_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_missed_call_operation_receipts_key
        CHECK (length(btrim(operation_key)) BETWEEN 16 AND 128),
    CONSTRAINT ck_eom_missed_call_operation_receipts_kind
        CHECK (operation_kind IN ('no_answer', 'resume', 'cancel')),
    CONSTRAINT ck_eom_missed_call_operation_receipts_fingerprint
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS eom_missed_call_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    operation_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    actor_id BIGINT NOT NULL,
    actor_name VARCHAR(128) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'time_tracker',
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_missed_call_attempts_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_missed_call_attempts_operation_key
        CHECK (length(btrim(operation_key)) BETWEEN 16 AND 128),
    CONSTRAINT ck_eom_missed_call_attempts_fingerprint
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_eom_missed_call_attempts_actor
        CHECK (actor_id > 0 AND length(btrim(actor_name)) > 0),
    CONSTRAINT ck_eom_missed_call_attempts_source
        CHECK (source = 'time_tracker'),
    CONSTRAINT uq_eom_missed_call_attempt_operation
        UNIQUE (operation_key)
);

CREATE TABLE IF NOT EXISTS eom_missed_call_contact_suppressions (
    contact_id UUID PRIMARY KEY REFERENCES contacts(id) ON DELETE RESTRICT,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    reason_code VARCHAR(64) NOT NULL,
    actor_id BIGINT,
    actor_name VARCHAR(128) NOT NULL DEFAULT 'system',
    source VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_missed_call_suppressions_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_missed_call_suppressions_reason
        CHECK (reason_code IN ('opt_out', 'unsubscribed', 'do_not_contact')),
    CONSTRAINT ck_eom_missed_call_suppressions_actor
        CHECK (length(btrim(actor_name)) > 0),
    CONSTRAINT ck_eom_missed_call_suppressions_source
        CHECK (source IN ('time_tracker', 'interaction_trigger'))
);

CREATE TABLE IF NOT EXISTS eom_missed_call_sequences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    initiating_attempt_id UUID NOT NULL
        REFERENCES eom_missed_call_attempts(id) ON DELETE RESTRICT,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    recipient_email VARCHAR(256) NOT NULL,
    state VARCHAR(32) NOT NULL,
    blocked_reason VARCHAR(64),
    cancellation_reason VARCHAR(64),
    terminal_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_missed_call_sequences_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT uq_eom_missed_call_sequences_initiating_attempt
        UNIQUE (initiating_attempt_id),
    CONSTRAINT ck_eom_missed_call_sequences_state
        CHECK (state IN (
            'active', 'blocked_configuration', 'completed', 'cancelled',
            'failed', 'recovery_required'
        )),
    CONSTRAINT ck_eom_missed_call_sequences_terminal_shape
        CHECK (
            (state IN ('active', 'blocked_configuration') AND terminal_at IS NULL)
            OR (state NOT IN ('active', 'blocked_configuration') AND terminal_at IS NOT NULL)
        ),
    CONSTRAINT ck_eom_missed_call_sequences_reason_shape
        CHECK (
            (state = 'blocked_configuration' AND blocked_reason IS NOT NULL)
            OR (state <> 'blocked_configuration' AND blocked_reason IS NULL)
        )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_eom_missed_call_sequences_active_contact
    ON eom_missed_call_sequences (contact_id)
    WHERE state IN ('active', 'blocked_configuration');

CREATE INDEX IF NOT EXISTS idx_eom_missed_call_sequences_contact
    ON eom_missed_call_sequences (contact_id, created_at DESC);

CREATE TABLE IF NOT EXISTS eom_missed_call_sequence_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID NOT NULL REFERENCES eom_missed_call_sequences(id)
        ON DELETE RESTRICT,
    step_number SMALLINT NOT NULL,
    due_at TIMESTAMPTZ NOT NULL,
    subject VARCHAR(500) NOT NULL,
    body TEXT NOT NULL,
    provider_idempotency_key VARCHAR(192) NOT NULL,
    provider_key_expires_at TIMESTAMPTZ,
    state VARCHAR(32) NOT NULL DEFAULT 'pending',
    attempt_count SMALLINT NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ,
    claim_token UUID,
    claimed_at TIMESTAMPTZ,
    claim_expires_at TIMESTAMPTZ,
    provider_message_id VARCHAR(256),
    sent_at TIMESTAMPTZ,
    terminal_reason VARCHAR(64),
    last_error_code VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_eom_missed_call_sequence_step
        UNIQUE (sequence_id, step_number),
    CONSTRAINT uq_eom_missed_call_step_provider_key
        UNIQUE (provider_idempotency_key),
    CONSTRAINT ck_eom_missed_call_step_number
        CHECK (step_number BETWEEN 1 AND 3),
    CONSTRAINT ck_eom_missed_call_step_state
        CHECK (state IN (
            'pending', 'attempting', 'sent', 'skipped', 'failed',
            'recovery_required'
        )),
    CONSTRAINT ck_eom_missed_call_step_attempt_count
        CHECK (attempt_count >= 0 AND attempt_count <= 5),
    CONSTRAINT ck_eom_missed_call_step_pending_shape
        CHECK (
            (state = 'pending' AND sent_at IS NULL AND provider_message_id IS NULL)
            OR state <> 'pending'
        ),
    CONSTRAINT ck_eom_missed_call_step_sent_shape
        CHECK (
            (state = 'sent' AND sent_at IS NOT NULL
             AND length(btrim(COALESCE(provider_message_id, ''))) > 0)
            OR (state <> 'sent' AND sent_at IS NULL AND provider_message_id IS NULL)
        ),
    -- A claim has to be durable and complete before a worker may call Resend.
    -- This makes an impossible half-claimed state reject at the same boundary
    -- that owns the provider idempotency window.
    CONSTRAINT ck_eom_missed_call_step_claim_shape
        CHECK (
            (state = 'attempting'
             AND claim_token IS NOT NULL
             AND claimed_at IS NOT NULL
             AND claim_expires_at IS NOT NULL)
            OR (state <> 'attempting'
                AND claim_token IS NULL
                AND claimed_at IS NULL
                AND claim_expires_at IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS idx_eom_missed_call_steps_due
    ON eom_missed_call_sequence_steps (due_at, next_attempt_at, created_at)
    WHERE state IN ('pending', 'attempting');

CREATE TABLE IF NOT EXISTS eom_missed_call_sequence_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID NOT NULL REFERENCES eom_missed_call_sequences(id)
        ON DELETE RESTRICT,
    step_id UUID REFERENCES eom_missed_call_sequence_steps(id) ON DELETE RESTRICT,
    event_type VARCHAR(64) NOT NULL,
    reason_code VARCHAR(64),
    actor_id BIGINT,
    actor_name VARCHAR(128) NOT NULL DEFAULT 'system',
    source VARCHAR(32) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_missed_call_events_actor
        CHECK (length(btrim(actor_name)) > 0),
    CONSTRAINT ck_eom_missed_call_events_source
        CHECK (source IN ('time_tracker', 'worker', 'contact_trigger', 'interaction_trigger')),
    CONSTRAINT ck_eom_missed_call_events_type
        CHECK (event_type IN (
            'sequence_started', 'sequence_reused', 'sequence_blocked',
            'sequence_resumed', 'sequence_cancelled', 'sequence_completed',
            'step_claimed', 'step_sent', 'step_retry_scheduled', 'step_skipped',
            'step_failed', 'step_recovery_required'
        ))
);

CREATE INDEX IF NOT EXISTS idx_eom_missed_call_sequence_events_sequence
    ON eom_missed_call_sequence_events (sequence_id, occurred_at DESC, id DESC);

CREATE OR REPLACE FUNCTION prevent_eom_missed_call_attempt_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom_missed_call_attempts is append-only';
END;
$$;

CREATE OR REPLACE FUNCTION prevent_eom_missed_call_operation_receipt_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom_missed_call_operation_receipts is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_operation_receipt_mutation
    ON eom_missed_call_operation_receipts;
CREATE TRIGGER trg_prevent_eom_missed_call_operation_receipt_mutation
    BEFORE UPDATE OR DELETE ON eom_missed_call_operation_receipts
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_missed_call_operation_receipt_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_operation_receipt_truncate
    ON eom_missed_call_operation_receipts;
CREATE TRIGGER trg_prevent_eom_missed_call_operation_receipt_truncate
    BEFORE TRUNCATE ON eom_missed_call_operation_receipts
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_missed_call_operation_receipt_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_attempt_mutation
    ON eom_missed_call_attempts;
CREATE TRIGGER trg_prevent_eom_missed_call_attempt_mutation
    BEFORE UPDATE OR DELETE ON eom_missed_call_attempts
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_missed_call_attempt_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_attempt_truncate
    ON eom_missed_call_attempts;
CREATE TRIGGER trg_prevent_eom_missed_call_attempt_truncate
    BEFORE TRUNCATE ON eom_missed_call_attempts
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_missed_call_attempt_mutation();

CREATE OR REPLACE FUNCTION prevent_eom_missed_call_sequence_event_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom_missed_call_sequence_events is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_sequence_event_mutation
    ON eom_missed_call_sequence_events;
CREATE TRIGGER trg_prevent_eom_missed_call_sequence_event_mutation
    BEFORE UPDATE OR DELETE ON eom_missed_call_sequence_events
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_missed_call_sequence_event_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_sequence_event_truncate
    ON eom_missed_call_sequence_events;
CREATE TRIGGER trg_prevent_eom_missed_call_sequence_event_truncate
    BEFORE TRUNCATE ON eom_missed_call_sequence_events
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_missed_call_sequence_event_mutation();

CREATE OR REPLACE FUNCTION prevent_eom_missed_call_suppression_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom_missed_call_contact_suppressions is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_suppression_mutation
    ON eom_missed_call_contact_suppressions;
CREATE TRIGGER trg_prevent_eom_missed_call_suppression_mutation
    BEFORE UPDATE OR DELETE ON eom_missed_call_contact_suppressions
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_missed_call_suppression_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_missed_call_suppression_truncate
    ON eom_missed_call_contact_suppressions;
CREATE TRIGGER trg_prevent_eom_missed_call_suppression_truncate
    BEFORE TRUNCATE ON eom_missed_call_contact_suppressions
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_missed_call_suppression_mutation();

CREATE OR REPLACE FUNCTION validate_eom_missed_call_contact_scope()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM contacts
         WHERE id = NEW.contact_id
           AND business_context_id = 'effingham_maids'
    ) THEN
        RAISE EXCEPTION 'eom missed-call recovery rows require an EOM contact';
    END IF;

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION validate_eom_missed_call_sequence_scope()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM contacts
         WHERE id = NEW.contact_id
           AND business_context_id = 'effingham_maids'
    ) THEN
        RAISE EXCEPTION 'eom missed-call recovery rows require an EOM contact';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM eom_missed_call_attempts AS attempt
         WHERE attempt.id = NEW.initiating_attempt_id
           AND attempt.contact_id = NEW.contact_id
           AND attempt.business_context_id = 'effingham_maids'
    ) THEN
        RAISE EXCEPTION 'eom missed-call recovery sequence must match its initiating attempt';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_validate_eom_missed_call_attempt_scope
    ON eom_missed_call_attempts;
CREATE TRIGGER trg_validate_eom_missed_call_attempt_scope
    BEFORE INSERT OR UPDATE ON eom_missed_call_attempts
    FOR EACH ROW
    EXECUTE FUNCTION validate_eom_missed_call_contact_scope();

DROP TRIGGER IF EXISTS trg_validate_eom_missed_call_operation_receipt_scope
    ON eom_missed_call_operation_receipts;
CREATE TRIGGER trg_validate_eom_missed_call_operation_receipt_scope
    BEFORE INSERT OR UPDATE ON eom_missed_call_operation_receipts
    FOR EACH ROW
    EXECUTE FUNCTION validate_eom_missed_call_contact_scope();

DROP TRIGGER IF EXISTS trg_validate_eom_missed_call_suppression_scope
    ON eom_missed_call_contact_suppressions;
CREATE TRIGGER trg_validate_eom_missed_call_suppression_scope
    BEFORE INSERT OR UPDATE ON eom_missed_call_contact_suppressions
    FOR EACH ROW
    EXECUTE FUNCTION validate_eom_missed_call_contact_scope();

DROP TRIGGER IF EXISTS trg_validate_eom_missed_call_sequence_scope
    ON eom_missed_call_sequences;
CREATE TRIGGER trg_validate_eom_missed_call_sequence_scope
    BEFORE INSERT OR UPDATE ON eom_missed_call_sequences
    FOR EACH ROW
    EXECUTE FUNCTION validate_eom_missed_call_sequence_scope();

CREATE OR REPLACE FUNCTION cancel_eom_missed_call_sequences_for_contact(
    target_contact_id UUID,
    target_reason VARCHAR(64),
    target_source VARCHAR(32)
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    WITH cancelled AS (
        UPDATE eom_missed_call_sequences
           SET state = 'cancelled',
               blocked_reason = NULL,
               cancellation_reason = target_reason,
               terminal_at = CURRENT_TIMESTAMP,
               updated_at = CURRENT_TIMESTAMP
         WHERE contact_id = target_contact_id
           AND state IN ('active', 'blocked_configuration')
         RETURNING id
    ), skipped AS (
        UPDATE eom_missed_call_sequence_steps
           SET state = 'skipped',
               terminal_reason = target_reason,
               next_attempt_at = NULL,
               claim_token = NULL,
               claimed_at = NULL,
               claim_expires_at = NULL,
               updated_at = CURRENT_TIMESTAMP
         WHERE sequence_id IN (SELECT id FROM cancelled)
           AND state IN ('pending', 'attempting')
         RETURNING id, sequence_id
    )
    INSERT INTO eom_missed_call_sequence_events (
        sequence_id, step_id, event_type, reason_code, actor_name, source
    )
    SELECT id, NULL, 'sequence_cancelled', target_reason, 'system', target_source
      FROM cancelled
    UNION ALL
    SELECT sequence_id, id, 'step_skipped', target_reason, 'system', target_source
      FROM skipped;
END;
$$;

CREATE OR REPLACE FUNCTION eom_missed_call_effective_recipient(
    target_contact_id UUID,
    fallback_contact_email TEXT
)
RETURNS TEXT
LANGUAGE sql
STABLE
AS $$
    -- Keep this precedence equal to the delivery service: an estimate form's
    -- submitted address remains the recovery recipient until a later estimate
    -- request replaces it. A routine edit to contacts.email alone must not
    -- silently cancel a sequence that still targets that submitted address.
    SELECT lower(
        NULLIF(
            btrim(
                COALESCE(
                    (
                        SELECT NULLIF(ci.metadata->>'submitted_email', '')
                        FROM contact_interactions AS ci
                        WHERE ci.contact_id = target_contact_id
                          AND ci.interaction_type = 'web_form'
                          AND ci.intent = 'estimate_request'
                        ORDER BY ci.occurred_at DESC, ci.created_at DESC, ci.id DESC
                        LIMIT 1
                    ),
                    fallback_contact_email
                )
            ),
            ''
        )
    );
$$;

CREATE OR REPLACE FUNCTION cancel_eom_missed_call_on_contact_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    cancellation_reason VARCHAR(64);
BEGIN
    IF NEW.business_context_id IS DISTINCT FROM 'effingham_maids' THEN
        RETURN NEW;
    END IF;

    IF NEW.contact_type <> 'lead' THEN
        cancellation_reason := 'became_customer';
    ELSIF NEW.status <> 'active' THEN
        cancellation_reason := 'contact_inactive';
    ELSIF NEW.lead_stage IS DISTINCT FROM 'new' THEN
        cancellation_reason := 'lead_advanced';
    ELSIF NEW.customer_type = 'commercial' THEN
        cancellation_reason := 'non_residential';
    ELSIF NEW.email IS DISTINCT FROM OLD.email
       AND EXISTS (
           SELECT 1
           FROM eom_missed_call_sequences AS sequence
           WHERE sequence.contact_id = NEW.id
             AND sequence.state IN ('active', 'blocked_configuration')
             AND lower(btrim(sequence.recipient_email)) IS DISTINCT FROM
                 eom_missed_call_effective_recipient(NEW.id, NEW.email)
       ) THEN
        cancellation_reason := 'recipient_changed';
    ELSE
        RETURN NEW;
    END IF;

    PERFORM cancel_eom_missed_call_sequences_for_contact(
        NEW.id, cancellation_reason, 'contact_trigger'
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_cancel_eom_missed_call_on_contact_change ON contacts;
CREATE TRIGGER trg_cancel_eom_missed_call_on_contact_change
    AFTER UPDATE OF contact_type, status, lead_stage, customer_type, email ON contacts
    FOR EACH ROW
    EXECUTE FUNCTION cancel_eom_missed_call_on_contact_change();

CREATE OR REPLACE FUNCTION eom_missed_call_has_proven_inbound_sms(
    candidate_metadata JSONB
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
AS $$
    -- EOM's current inbound SMS bridge writes crm_event_id="sms:<provider id>".
    -- A future bridge may instead carry an explicit direction. Any other
    -- generic CRM `sms` interaction (including an outgoing reminder) is not
    -- evidence that this lead replied, so it intentionally does not cancel.
    SELECT COALESCE(
        candidate_metadata->>'direction' = 'inbound'
        OR NULLIF(btrim(candidate_metadata->>'crm_event_id'), '') LIKE 'sms:%',
        FALSE
    );
$$;

CREATE OR REPLACE FUNCTION cancel_eom_missed_call_on_interaction()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    cancellation_reason VARCHAR(64);
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM contacts
         WHERE id = NEW.contact_id
           AND business_context_id = 'effingham_maids'
    ) THEN
        RETURN NEW;
    END IF;

    IF NEW.metadata->>'missed_call_recovery_cancel_reason' IN (
        'callback_recorded', 'response_recorded', 'opt_out', 'manual'
    ) THEN
        -- The authenticated operator route records both the standard CRM
        -- interaction type and the operator's more precise stop reason. Keep
        -- that durable reason in the sequence ledger instead of flattening it
        -- to the generic interaction vocabulary.
        cancellation_reason := NEW.metadata->>'missed_call_recovery_cancel_reason';
    ELSIF NEW.interaction_type = 'sms'
       AND eom_missed_call_has_proven_inbound_sms(NEW.metadata) THEN
        cancellation_reason := 'tracked_inbound_response';
    ELSIF NEW.interaction_type IN (
        'email_inbound', 'lead_response', 'callback_completed',
        'conversation_completed', 'opt_out'
    ) THEN
        cancellation_reason := NEW.interaction_type;
    ELSIF NEW.interaction_type = 'web_form'
       AND NEW.intent = 'estimate_request' THEN
        cancellation_reason := 'new_estimate_request';
    ELSE
        RETURN NEW;
    END IF;

    IF NEW.interaction_type = 'opt_out' THEN
        INSERT INTO eom_missed_call_contact_suppressions (
            contact_id, reason_code, actor_name, source
        ) VALUES (
            NEW.contact_id, 'opt_out', 'system', 'interaction_trigger'
        ) ON CONFLICT (contact_id) DO NOTHING;
    END IF;

    -- A later-arriving historical event is evidence about an earlier point in
    -- time, not a new response after this sequence began. The delivery worker
    -- uses the same strict ordering when it rechecks current eligibility.
    IF EXISTS (
        SELECT 1
        FROM eom_missed_call_sequences AS sequence
        WHERE sequence.contact_id = NEW.contact_id
          AND sequence.state IN ('active', 'blocked_configuration')
          AND NEW.occurred_at > sequence.created_at
    ) THEN
        PERFORM cancel_eom_missed_call_sequences_for_contact(
            NEW.contact_id, cancellation_reason, 'interaction_trigger'
        );
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_cancel_eom_missed_call_on_interaction
    ON contact_interactions;
CREATE TRIGGER trg_cancel_eom_missed_call_on_interaction
    AFTER INSERT ON contact_interactions
    FOR EACH ROW
    EXECUTE FUNCTION cancel_eom_missed_call_on_interaction();

COMMENT ON TABLE eom_missed_call_attempts IS
    'Immutable, actor-attributed EOM no-answer call evidence; never inferred from public form submission.';
COMMENT ON TABLE eom_missed_call_operation_receipts IS
    'Globally unique EOM missed-call recovery operation-key ownership; cross-contact replays fail closed.';
COMMENT ON TABLE eom_missed_call_sequences IS
    'Current durable state for one EOM residential lead missed-call recovery sequence.';
COMMENT ON TABLE eom_missed_call_contact_suppressions IS
    'EOM-local durable do-not-contact evidence for future missed-call sequences.';
COMMENT ON TABLE eom_missed_call_sequence_steps IS
    'Deterministic outbox steps with bounded retry and Resend idempotency evidence.';
COMMENT ON TABLE eom_missed_call_sequence_events IS
    'Append-only EOM missed-call recovery transition and delivery evidence.';
