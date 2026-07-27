-- Durable replay receipts for EOM inbound deliveries.
--
-- contacts.source_ref is provenance for one contact origin, not a delivery
-- ledger: a known contact can receive many Web3Forms, call, or SMS deliveries.
-- This receipt records each trusted delivery exactly once and pins it to the
-- contact/interaction selected by the atomic ingress command.

CREATE TABLE IF NOT EXISTS eom_inbound_delivery_receipts (
    source VARCHAR(64) NOT NULL,
    delivery_id VARCHAR(256) NOT NULL,
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    interaction_id UUID REFERENCES contact_interactions(id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, delivery_id)
);

CREATE INDEX IF NOT EXISTS idx_eom_inbound_delivery_receipts_contact
    ON eom_inbound_delivery_receipts (contact_id, created_at DESC);

COMMENT ON TABLE eom_inbound_delivery_receipts IS
    'EOM trusted inbound-delivery replay ledger; maps one source/delivery ID to its resolved CRM contact and optional interaction.';
