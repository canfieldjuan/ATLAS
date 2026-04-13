-- Duplicate review rows should not carry independent vendor associations.
--
-- Canonical review-to-vendor lookups must hang off the survivor row only.

DELETE FROM b2b_review_vendor_mentions vm
USING b2b_reviews r
WHERE r.id = vm.review_id
  AND r.duplicate_of_review_id IS NOT NULL;
