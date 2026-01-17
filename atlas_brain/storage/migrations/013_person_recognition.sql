-- Atlas Brain Database Schema
-- Migration 013: Person Recognition (Face & Gait)
-- Enables facial recognition and gait analysis

-- Persons table - known individuals
CREATE TABLE IF NOT EXISTS persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    is_known BOOLEAN DEFAULT TRUE,
    auto_created BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);
CREATE INDEX IF NOT EXISTS idx_persons_is_known ON persons(is_known);
CREATE INDEX IF NOT EXISTS idx_persons_last_seen ON persons(last_seen_at DESC);

-- Face embeddings table (512-dim for InsightFace/ArcFace)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL,
    quality_score FLOAT DEFAULT 0.0,
    source VARCHAR(50) DEFAULT 'enrollment',
    reference_image BYTEA,
    image_format VARCHAR(10) DEFAULT 'jpeg',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_face_person_id ON face_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_face_embedding ON face_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Gait embeddings table (256-dim for gait signatures)
CREATE TABLE IF NOT EXISTS gait_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding vector(256) NOT NULL,
    capture_duration_ms INTEGER,
    frame_count INTEGER,
    walking_direction VARCHAR(20),
    source VARCHAR(50) DEFAULT 'enrollment',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_gait_person_id ON gait_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_gait_embedding ON gait_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Recognition events log
CREATE TABLE IF NOT EXISTS recognition_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    recognition_type VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    camera_source VARCHAR(100),
    matched BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_recog_person_id ON recognition_events(person_id);
CREATE INDEX IF NOT EXISTS idx_recog_type ON recognition_events(recognition_type);
CREATE INDEX IF NOT EXISTS idx_recog_created ON recognition_events(created_at DESC);

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (13, '013_person_recognition')
ON CONFLICT (version) DO NOTHING;
