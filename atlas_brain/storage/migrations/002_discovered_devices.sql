-- Atlas Brain Database Schema
-- Migration 002: Discovered Devices
-- Creates table for network-discovered devices

-- Discovered devices table for auto-discovered network devices
CREATE TABLE IF NOT EXISTS discovered_devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id VARCHAR(255) UNIQUE NOT NULL,  -- Unique identifier like "roku.192_168_1_2"
    name VARCHAR(255) NOT NULL,               -- Human-readable name
    device_type VARCHAR(50) NOT NULL,         -- "roku", "chromecast", "smart_tv", etc.
    protocol VARCHAR(50) NOT NULL,            -- Discovery protocol: "ssdp", "mdns", "manual"
    host VARCHAR(255) NOT NULL,               -- IP address or hostname
    port INTEGER,                             -- Port if applicable
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,           -- Currently reachable
    auto_registered BOOLEAN DEFAULT FALSE,    -- Auto-added to capability registry
    metadata JSONB DEFAULT '{}'::jsonb        -- Protocol-specific data (SSDP headers, etc.)
);

-- Indexes for device queries
CREATE INDEX IF NOT EXISTS idx_discovered_devices_device_id ON discovered_devices(device_id);
CREATE INDEX IF NOT EXISTS idx_discovered_devices_type ON discovered_devices(device_type);
CREATE INDEX IF NOT EXISTS idx_discovered_devices_active ON discovered_devices(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_discovered_devices_host ON discovered_devices(host);
CREATE INDEX IF NOT EXISTS idx_discovered_devices_last_seen ON discovered_devices(last_seen_at DESC);

-- Record this migration
INSERT INTO schema_migrations (version, name)
VALUES (2, '002_discovered_devices')
ON CONFLICT (version) DO NOTHING;
