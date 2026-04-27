-- Allow win_probability to be NULL on persisted predictions.
-- Gated predictions (insufficient data) cannot honestly assert a probability;
-- storing NULL avoids leaking fake numerics into history/compare/export.
--
-- The replacement invariant: a non-gated prediction MUST have a probability.
-- This prevents an "unmarked-but-null" middle state that would later crash
-- compare/history/export assumptions.

ALTER TABLE b2b_win_loss_predictions
ALTER COLUMN win_probability DROP NOT NULL;

ALTER TABLE b2b_win_loss_predictions
ADD CONSTRAINT b2b_win_loss_predictions_probability_invariant
CHECK (is_gated OR win_probability IS NOT NULL);
