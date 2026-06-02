-- Persist the delivery address captured during FAQ deflection report submit.

ALTER TABLE content_ops_deflection_reports
    ADD COLUMN IF NOT EXISTS delivery_email TEXT;
