import type { InsightPost } from "@/types";

export const edgeCloudVoicePipeline: InsightPost = {
  slug: "edge-cloud-voice-pipeline-60-dollar-board",
  title: "Splitting a Voice Pipeline Across a $60 ARM Board and a GPU Server",
  description:
    "How I architected an always-on voice assistant with STT and TTS on an Orange Pi edge node, LLM reasoning in the cloud, and sub-second local response for common commands.",
  date: "2026-03-22",
  type: "case-study",
  tags: [
    "edge compute",
    "voice pipeline",
    "ARM",
    "NPU",
    "latency optimization",
    "distributed systems",
  ],
  project: "atlas",
  seoTitle: "Edge/Cloud Voice Pipeline: ARM Board + GPU Server Architecture",
  seoDescription:
    "Case study: splitting voice AI across an Orange Pi RK3588 ($60) and a GPU server. STT, TTS, and computer vision on-device. LLM reasoning in the cloud. Sub-second local skills.",
  targetKeyword: "edge cloud ai voice pipeline",
  secondaryKeywords: [
    "orange pi ai",
    "arm npu voice assistant",
    "distributed ai architecture",
  ],
  content: `
<h2>The Architecture Question</h2>
<p>Building a voice assistant means choosing: cloud-only (high latency, simple), edge-only (limited capability), or split (complex, but right). I chose split — and the complexity is where the interesting engineering lives.</p>

<h2>Edge Node: Orange Pi RK3588</h2>
<p>The edge node is an Orange Pi with an RK3588 SoC — 8 ARM cores, 6 TOPS NPU, 8GB RAM, $60 total cost. It runs:</p>
<ul>
  <li><strong>STT:</strong> SenseVoice int8 ONNX via sherpa-onnx (CPU, 2 threads). Fast enough for real-time transcription without touching the NPU.</li>
  <li><strong>TTS:</strong> Piper (en_US-amy-low model). ~6.6x realtime on the RK3588 — generating speech 6.6x faster than playback speed.</li>
  <li><strong>Computer Vision:</strong> YOLO-World (NPU core 0), RetinaFace (core 1), MobileFaceNet (core 2), YOLOv8n-pose (core 1, timeshared). Motion gate (MOG2 on CPU) prevents NPU inference when nothing's moving.</li>
  <li><strong>Local Skills:</strong> Time, timer, math, status queries skip the brain entirely. Sub-second response.</li>
</ul>

<p>Total memory footprint: ~1.7GB for the Atlas node, ~400MB for Home Assistant (also running on-device), plus Docker containers for Pi-hole and MediaMTX (RTSP streaming).</p>

<h2>Brain: GPU Server</h2>
<p>The brain runs on a separate machine with an RTX 3090. It handles:</p>
<ul>
  <li><strong>LLM reasoning:</strong> Qwen3:14b via Ollama (~10GB VRAM). Conversation, intent classification, tool calling.</li>
  <li><strong>ASR server:</strong> Nemotron 0.6B for high-quality batch transcription (~2GB VRAM).</li>
  <li><strong>All persistence:</strong> PostgreSQL, Neo4j, conversation history, device state.</li>
  <li><strong>MCP servers:</strong> 11 servers, 190 tools — all heavy compute stays here.</li>
</ul>

<h2>The Connection: Tailscale</h2>
<p>Edge and brain communicate over Tailscale (WireGuard mesh VPN). WebSocket for real-time streaming, HTTP for batch operations. The connection is encrypted, traverses NATs, and adds ~2ms latency on the local network.</p>

<h2>Optimization Lessons (The Tedious Parts)</h2>

<h3>Sequential Dispatch Was Killing Latency</h3>
<p>The first version dispatched operations sequentially — send audio, wait for transcription, send text, wait for response, send TTS. Fixed with <code>_spawn_task()</code> background tasks. Operations that don't depend on each other run concurrently.</p>

<h3>Per-Token JSON Overhead</h3>
<p>Streaming LLM responses token-by-token over WebSocket meant wrapping every 2-3 character token in a JSON envelope. The overhead was larger than the payload. Fixed with a <code>TokenBatcher</code> class that buffers tokens and sends them in batches.</p>

<h3>Per-Detection Sequential DB Writes</h3>
<p>Computer vision was writing each detection to the database individually. 10 detections = 10 round trips. Fixed with <code>executemany</code> batch inserts.</p>

<h3>Lazy Imports in the Hot Path</h3>
<p>Python's import system is not free. Lazy imports inside request handlers added 50-100ms on first call. Fixed by moving all imports to module level.</p>

<h3>No WebSocket Compression</h3>
<p>Audio chunks and LLM responses are compressible. Added app-level zlib compression. Edge opts in via <code>capabilities.compression: "zlib"</code> during WebSocket handshake. The brain checks the capability before compressing.</p>

<h2>NPU Core Allocation</h2>
<p>The RK3588 has 3 NPU cores. Naively loading all models on "the NPU" causes contention. The allocation:</p>
<ul>
  <li><strong>Core 0:</strong> YOLO-World (object detection) — highest priority, always-on when motion detected</li>
  <li><strong>Core 1:</strong> RetinaFace (face detection) + YOLOv8n-pose (timeshared) — face detection runs first, pose runs in gaps</li>
  <li><strong>Core 2:</strong> MobileFaceNet (face recognition) — triggered only when RetinaFace finds a face</li>
</ul>
<p>The motion gate is critical: MOG2 background subtraction runs on CPU and prevents any NPU inference when nothing's moving. This saves power and prevents the NPU from running hot 24/7.</p>

<h2>What This Architecture Enables</h2>
<p>Common commands ("Hey Atlas, turn off the TV") resolve entirely on the edge node — intent matched locally, Home Assistant command sent locally, TTS response generated locally. Round trip: under 1 second.</p>
<p>Complex queries ("What's on my calendar tomorrow and should I reschedule the dentist?") route to the brain for LLM reasoning + tool calling, then stream TTS back to the edge. Round trip: 3-5 seconds depending on LLM response length.</p>
<p>The user doesn't know or care about the split. That's the point.</p>
`,
};
