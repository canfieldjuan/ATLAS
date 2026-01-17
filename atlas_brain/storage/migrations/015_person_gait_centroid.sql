-- Atlas Brain Database Schema
-- Migration 015: Add gait centroid embedding to persons
-- Enables O(1) pgvector search for gait recognition

-- Add gait centroid embedding column to persons table
ALTER TABLE persons
ADD COLUMN IF NOT EXISTS gait_centroid vector(256),
ADD COLUMN IF NOT EXISTS gait_sample_count INTEGER DEFAULT 0;

-- Create index for fast similarity search on gait centroids
CREATE INDEX IF NOT EXISTS idx_persons_gait_centroid ON persons
USING ivfflat (gait_centroid vector_cosine_ops)
WITH (lists = 100);

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (15, '015_person_gait_centroid')
ON CONFLICT (version) DO NOTHING;
