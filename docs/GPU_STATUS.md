# GPU Status

> Last checked: 2026-04-11

## Current State: GPU NOT DETECTED

The NVIDIA driver (580.126.09) loads but reports "No NVIDIA GPU found."
The kernel log shows repeated failed initialization attempts.

### Impact

All tasks designed for local vLLM inference (Qwen3-30B-A3B-AWQ) are
falling back to paid Anthropic/OpenRouter APIs. This is the primary
cause of elevated API costs.

### Affected Tasks

| Task | Designed For | Falling Back To |
|---|---|---|
| b2b_churn_intelligence | vLLM (free) | Anthropic triage (paid) |
| Atlas Agent | vLLM (free) | Non-functional |
| Home Agent (voice) | vLLM (free) | Non-functional |
| Intent router LLM | Ollama (free) | Non-functional |
| Email graph sync | Ollama (free) | Non-functional |
| deep_enrichment | vLLM (free) | Disabled |
| complaint_enrichment | vLLM (free) | Disabled |

### Recovery

1. Full power cycle the PC (power off at PSU, wait 30s, power on)
2. After boot, verify: `nvidia-smi`
3. If GPU detected, start vLLM: `cd ~/Desktop/Atlas && source .venv/bin/activate && nohup vllm serve stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ --port 8001 > /tmp/vllm.log 2>&1 &`
4. Start Ollama: `ollama serve` or `systemctl start ollama`
5. Re-enable local-inference tasks as needed
