-- Normalize CRM push success status semantics.

ALTER TABLE b2b_crm_push_log
    ALTER COLUMN status SET DEFAULT 'success';

UPDATE b2b_crm_push_log
SET status = 'success'
WHERE status = 'pushed';
