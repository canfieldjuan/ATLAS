-- Add llm_model column to b2b_cross_vendor_conclusions for MCP client provenance.
-- Pipeline rows keep NULL; MCP-client rows use 'mcp-client:<model>' convention.
ALTER TABLE b2b_cross_vendor_conclusions
    ADD COLUMN IF NOT EXISTS llm_model TEXT;
