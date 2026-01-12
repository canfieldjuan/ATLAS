-- Atlas Brain Database Schema
-- Migration 004: User Profiles
-- Adds personalization settings for users

-- User profiles table for personalization
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    display_name VARCHAR(100),
    timezone VARCHAR(50) DEFAULT 'UTC',
    locale VARCHAR(20) DEFAULT 'en-US',
    response_style VARCHAR(20) DEFAULT 'balanced',
    expertise_level VARCHAR(20) DEFAULT 'intermediate',
    enable_rag BOOLEAN DEFAULT TRUE,
    enable_context_injection BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Index for user profile lookup
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);

-- User preferences table for custom settings
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    preference_key VARCHAR(100) NOT NULL,
    preference_value TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, preference_key)
);

-- Index for preference lookup
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_key
    ON user_preferences(user_id, preference_key);

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (4, '004_user_profiles')
ON CONFLICT (version) DO NOTHING;
