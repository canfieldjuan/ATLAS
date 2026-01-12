-- Atlas Brain Database Schema
-- Migration 001: Initial Schema
-- Creates core tables for conversation persistence

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table for registered speakers
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    speaker_embedding BYTEA  -- Voice fingerprint for speaker ID
);

-- Index for user lookup by name
CREATE INDEX IF NOT EXISTS idx_users_name ON users(name);

-- Sessions table for active conversations
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    terminal_id VARCHAR(255),  -- Device/location identifier
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for session queries
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_terminal_id ON sessions(terminal_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity_at DESC);

-- Conversation turns table
CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    speaker_id VARCHAR(255),  -- Identified speaker name
    intent VARCHAR(255),  -- Parsed intent
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for conversation queries
CREATE INDEX IF NOT EXISTS idx_turns_session_id ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_created_at ON conversation_turns(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_turns_session_created ON conversation_turns(session_id, created_at DESC);

-- Terminals table for registered devices/locations
CREATE TABLE IF NOT EXISTS terminals (
    id VARCHAR(255) PRIMARY KEY,  -- User-defined like "office", "car", "home"
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    capabilities TEXT[] DEFAULT '{}',
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (1, '001_initial_schema')
ON CONFLICT (version) DO NOTHING;
