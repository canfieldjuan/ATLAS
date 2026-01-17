-- Atlas Brain Database Schema
-- Migration 014: Add face centroid embedding to persons
-- Enables O(1) pgvector search instead of O(n) Python iteration

-- Add centroid embedding column to persons table
ALTER TABLE persons
ADD COLUMN IF NOT EXISTS face_centroid vector(512),
ADD COLUMN IF NOT EXISTS face_sample_count INTEGER DEFAULT 0;

-- Create index for fast similarity search on centroids
CREATE INDEX IF NOT EXISTS idx_persons_face_centroid ON persons
USING ivfflat (face_centroid vector_cosine_ops)
WITH (lists = 100);

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (14, '014_person_face_centroid')
ON CONFLICT (version) DO NOTHING;
