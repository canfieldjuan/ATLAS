-- Migration 019: Add CHECK constraints for enum-like columns
-- Prevents invalid values at the DB level (belt-and-suspenders with API validation)

ALTER TABLE scheduled_tasks
  ADD CONSTRAINT chk_task_type CHECK (task_type IN ('agent_prompt', 'builtin', 'hook'));

ALTER TABLE scheduled_tasks
  ADD CONSTRAINT chk_schedule_type CHECK (schedule_type IN ('cron', 'interval', 'once'));

ALTER TABLE task_executions
  ADD CONSTRAINT chk_exec_status CHECK (status IN ('running', 'completed', 'failed', 'timeout'));

INSERT INTO schema_migrations (version, name)
VALUES (19, '019_check_constraints') ON CONFLICT (version) DO NOTHING;
