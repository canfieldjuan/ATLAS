-- Atlas Brain Database Schema
-- Migration 018: Scheduled Tasks (Autonomous Agent Execution)
-- Stores task definitions and execution history for the autonomous scheduler.

CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    task_type VARCHAR(64) NOT NULL,
    prompt TEXT,
    agent_type VARCHAR(32) DEFAULT 'atlas',
    schedule_type VARCHAR(16) NOT NULL,
    cron_expression VARCHAR(128),
    interval_seconds INTEGER,
    run_at TIMESTAMPTZ,
    timezone VARCHAR(64) DEFAULT 'America/Chicago',
    enabled BOOLEAN DEFAULT TRUE,
    max_retries INTEGER DEFAULT 0,
    retry_delay_seconds INTEGER DEFAULT 60,
    timeout_seconds INTEGER DEFAULT 120,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
    status VARCHAR(16) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    result_text TEXT,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_sched_tasks_enabled ON scheduled_tasks(enabled) WHERE enabled = TRUE;
CREATE INDEX IF NOT EXISTS idx_task_exec_task_id ON task_executions(task_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_exec_cleanup ON task_executions(completed_at) WHERE status IN ('completed','failed','timeout');

INSERT INTO schema_migrations (version, name)
VALUES (18, '018_scheduled_tasks')
ON CONFLICT (version) DO NOTHING;
