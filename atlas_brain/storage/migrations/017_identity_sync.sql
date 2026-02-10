-- Atlas Brain Database Schema
-- Migration 017: Identity Sync (Edge <-> Brain embedding distribution)
-- Stores face/gait/speaker embeddings by (name, modality) for multi-node sync.

-- Identity embeddings table -- master registry for edge node sync.
-- Each row is one embedding keyed by (name, modality).
-- Edge nodes store these as {name}.npy files in face_db/gait_db/speaker_db.
CREATE TABLE IF NOT EXISTS identity_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    modality VARCHAR(20) NOT NULL CHECK (modality IN ('face', 'gait', 'speaker')),
    embedding BYTEA NOT NULL,
    embedding_dim INTEGER NOT NULL,
    source_node VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name, modality)
);

CREATE INDEX IF NOT EXISTS idx_identity_name ON identity_embeddings(name);
CREATE INDEX IF NOT EXISTS idx_identity_modality ON identity_embeddings(modality);

-- Add speaker centroid to persons for completeness (face + gait already there)
ALTER TABLE persons
ADD COLUMN IF NOT EXISTS speaker_centroid vector(192),
ADD COLUMN IF NOT EXISTS speaker_sample_count INTEGER DEFAULT 0;

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (17, '017_identity_sync')
ON CONFLICT (version) DO NOTHING;
