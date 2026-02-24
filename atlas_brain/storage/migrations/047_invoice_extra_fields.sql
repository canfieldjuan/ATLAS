-- Add invoice_for and contact_name to invoices for template rendering
ALTER TABLE invoices ADD COLUMN IF NOT EXISTS invoice_for VARCHAR(256);
ALTER TABLE invoices ADD COLUMN IF NOT EXISTS contact_name VARCHAR(256);
