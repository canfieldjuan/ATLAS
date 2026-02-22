 Atlas System Audit

  1. Brain Modules (20 directories)

  Module: agents/
  Status: Active
  Purpose: LangGraph agent graphs (atlas, home, booking, calendar, email, reminder, security,
    presence, receptionist, streaming)
  ────────────────────────────────────────
  Module: alerts/
  Status: Active
  Purpose: AlertManager, rules engine, delivery (ntfy, TTS, DB), PresenceAlertEvent
  ────────────────────────────────────────
  Module: api/
  Status: Active
  Purpose: 18 REST routers (health, query, devices, autonomous, presence, actions, edge WS,
    identity, etc.)
  ────────────────────────────────────────
  Module: autonomous/
  Status: Active
  Purpose: Scheduler, hooks, event queue, presence tracker, 6 builtin tasks
  ────────────────────────────────────────
  Module: capabilities/
  Status: Active
  Purpose: Device control: HA REST + WS backends, MQTT, device resolver, state cache, intent
    parser
  ────────────────────────────────────────
  Module: comms/
  Status: Active
  Purpose: Phone/PersonaPlex processors, tool_bridge, real_services (Resend email, SignalWire SMS,
    Google Calendar). 13 API endpoints (voice webhooks, SMS, management). PersonaPlex speech-to-speech
    is primary call mode. Phone V2 STT/TTS pending V3 voice pipeline re-integration.
  ────────────────────────────────────────
  Module: discovery/
  Status: Active
  Purpose: SSDP + mDNS network device scanners
  ────────────────────────────────────────
  Module: jobs/
  Status: Active
  Purpose: nightly_memory_sync (now runs as autonomous builtin)
  ────────────────────────────────────────
  Module: memory/
  Status: Active
  Purpose: RAG client (Graphiti), query classifier, feedback loop, token estimator
  ────────────────────────────────────────
  Module: modes/
  Status: Active
  Purpose: Mode manager (HOME/AWAY/etc. with timeout fallback)
  ────────────────────────────────────────
  Module: orchestration/
  Status: Minimal
  Purpose: Just context.py — mostly superseded by LangGraph
  ────────────────────────────────────────
  Module: presence/
  Status: Stub
  Purpose: proxy.py for atlas_vision service, separate from autonomous/presence.py
  ────────────────────────────────────────
  Module: schemas/
  Status: Active
  Purpose: Pydantic query models
  ────────────────────────────────────────
  Module: services/
  Status: Active
  Purpose: LLM backends (5), embedding, intent router, speaker ID, tool executor,
    reminders, tracing
  ────────────────────────────────────────
  Module: storage/
  Status: Active
  Purpose: asyncpg pool, 12 repositories, models, migrations
  ────────────────────────────────────────
  Module: templates/
  Status: Active
  Purpose: Email templates (estimate confirmation, proposal)
  ────────────────────────────────────────
  Module: tools/
  Status: Active
  Purpose: 15 tool implementations (see below)
  ────────────────────────────────────────
  Module: vision/
  Status: Active
  Purpose: Vision event models + subscriber
  ────────────────────────────────────────
  Module: voice/
  Status: Active
  Purpose: Full voice pipeline: audio capture, VAD (Silero), segmenter, Kokoro TTS, playback

  ---
  2. Registered Tools (in tool_registry)

  Parameterless (fast-path, no LLM needed)

  ┌────────────────┬─────────────┬───────────────────────┐
  │      Tool      │    File     │   Qwen3 Difficulty    │
  ├────────────────┼─────────────┼───────────────────────┤
  │ get_time       │ time.py     │ N/A - no LLM call     │
  ├────────────────┼─────────────┼───────────────────────┤
  │ get_weather    │ weather.py  │ N/A - no LLM call     │
  ├────────────────┼─────────────┼───────────────────────┤
  │ get_calendar   │ calendar.py │ N/A - read-only query │
  ├────────────────┼─────────────┼───────────────────────┤
  │ list_reminders │ reminder.py │ N/A - read-only query │
  ├────────────────┼─────────────┼───────────────────────┤
  │ get_traffic    │ traffic.py  │ N/A - no LLM call     │
  ├────────────────┼─────────────┼───────────────────────┤
  │ get_location   │ location.py │ N/A - no LLM call     │
  ├────────────────┼─────────────┼───────────────────────┤
  │ where_am_i     │ presence.py │ N/A - no LLM call     │
  └────────────────┴─────────────┴───────────────────────┘

  These tools execute directly from the intent router with zero LLM involvement. The semantic
  router (~5ms) classifies intent, then the tool runs.

  Device/Presence Tools (registered, fast-path capable)

  ┌───────────────────┬─────────────┬─────────────────────────────┐
  │       Tool        │    File     │           Purpose           │
  ├───────────────────┼─────────────┼─────────────────────────────┤
  │ lights_near_user  │ presence.py │ Context-aware light control │
  ├───────────────────┼─────────────┼─────────────────────────────┤
  │ media_near_user   │ presence.py │ Context-aware media control │
  ├───────────────────┼─────────────┼─────────────────────────────┤
  │ scene_near_user   │ presence.py │ Room scene activation       │
  ├───────────────────┼─────────────┼─────────────────────────────┤
  │ send_notification │ notify.py   │ Push notification via ntfy  │
  └───────────────────┴─────────────┴─────────────────────────────┘

  Security Tools (registered)

  ┌──────────────────────────────────┬─────────────┬────────────────────────────┐
  │               Tool               │    File     │          Purpose           │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ list_cameras                     │ security.py │ List all cameras           │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ get_camera_status                │ security.py │ Check specific camera      │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ start_recording / stop_recording │ security.py │ Recording control          │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ ptz_control                      │ security.py │ Pan/tilt/zoom              │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ get_current_detections           │ security.py │ Current vision detections  │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ query_detections                 │ security.py │ Historical detection query │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ get_person_at_location           │ security.py │ Person lookup              │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ get_motion_events                │ security.py │ Motion history             │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ list_zones / get_zone_status     │ security.py │ Zone info                  │
  ├──────────────────────────────────┼─────────────┼────────────────────────────┤
  │ arm_zone / disarm_zone           │ security.py │ Security system control    │
  └──────────────────────────────────┴─────────────┴────────────────────────────┘

  Display Tools (registered)

  ┌───────────────────┬────────────┬────────────────────────┐
  │       Tool        │    File    │        Purpose         │
  ├───────────────────┼────────────┼────────────────────────┤
  │ show_camera_feed  │ display.py │ Show camera on monitor │
  ├───────────────────┼────────────┼────────────────────────┤
  │ close_camera_feed │ display.py │ Close camera viewer    │
  └───────────────────┴────────────┴────────────────────────┘

  Workflow Tools (registered, used by LLM tool-calling)

  ┌───────────────────────┬───────────────┬──────────┬────────────────────────────┐
  │         Tool          │     File      │ Workflow │      Qwen3 Difficulty      │
  ├───────────────────────┼───────────────┼──────────┼────────────────────────────┤
  │ set_reminder          │ reminder.py   │ reminder │ Easy - simple slot fill    │
  ├───────────────────────┼───────────────┼──────────┼────────────────────────────┤
  │ complete_reminder     │ reminder.py   │ reminder │ Easy                       │
  ├───────────────────────┼───────────────┼──────────┼────────────────────────────┤
  │ create_calendar_event │ calendar.py   │ calendar │ Medium - date/time parsing │
  ├───────────────────────┼───────────────┼──────────┼────────────────────────────┤
  │ check_availability    │ scheduling.py │ booking  │ Medium - multi-step        │
  ├───────────────────────┼───────────────┼──────────┼────────────────────────────┤
  │ book_appointment      │ scheduling.py │ booking  │ Medium - multi-step        │
  ├───────────────────────┼───────────────┼──────────┼────────────────────────────┤
  │ lookup_customer       │ scheduling.py │ booking  │ Easy                       │
  └───────────────────────┴───────────────┴──────────┴────────────────────────────┘

  NOT Registered (routed through workflows only)

  ┌────────────────────────┬───────────────┬─────────────────────────────────────┐
  │          Tool          │     File      │                 Why                 │
  ├────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ send_email             │ email.py      │ Routes through email workflow graph │
  ├────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ send_estimate_email    │ email.py      │ Business-specific template          │
  ├────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ send_proposal_email    │ email.py      │ Business-specific template          │
  └────────────────────────┴───────────────┴─────────────────────────────────────┘

  NOTE: cancel_appointment and reschedule_appointment are now registered and routed
  through the booking workflow via cancel_booking and reschedule_booking intent routes.

  ---
  3. Agent Graphs (LangGraph Workflows)

  Graph: atlas
  File: atlas.py
  Trigger: Main router - all queries enter here
  Qwen3 Difficulty: Classification only
  ────────────────────────────────────────
  Graph: home
  File: home.py
  Trigger: Device commands via delegate_home
  Qwen3 Difficulty: Easy - intent parse + HA call
  ────────────────────────────────────────
  Graph: reminder
  File: reminder.py
  Trigger: "remind me to..."
  Qwen3 Difficulty: Easy - 1-2 turn slot fill
  ────────────────────────────────────────
  Graph: calendar
  File: calendar.py
  Trigger: "add to my calendar..."
  Qwen3 Difficulty: Medium - date parsing
  ────────────────────────────────────────
  Graph: booking
  File: booking.py
  Trigger: "book an appointment..."
  Qwen3 Difficulty: Medium - multi-turn, 3-4 steps
  ────────────────────────────────────────
  Graph: email
  File: email.py
  Trigger: "send an email..."
  Qwen3 Difficulty: Hard - compose body, multi-turn
  ────────────────────────────────────────
  Graph: security
  File: security.py
  Trigger: Camera/zone queries
  Qwen3 Difficulty: Medium - tool selection
  ────────────────────────────────────────
  Graph: presence
  File: presence.py
  Trigger: "movie mode", "cozy lighting"
  Qwen3 Difficulty: Easy - scene mapping
  ────────────────────────────────────────
  Graph: receptionist
  File: receptionist.py
  Trigger: Inbound calls/visitors
  Qwen3 Difficulty: Hard - multi-tool, judgment
  ────────────────────────────────────────
  Graph: streaming
  File: streaming.py
  Trigger: Token streaming for edge
  Qwen3 Difficulty: N/A - infrastructure

  ---
  4. Cloud Routing Architecture

  Where It Lives

  services/llm/cloud.py — @register_llm("cloud") composite backend:
  - Primary: Groq API (llama-3.3-70b-versatile) — lowest latency
  - Fallback: Together.ai (Llama-3.3-70B-Instruct-Turbo) — auto-retry on Groq failure
  - Config: GROQ_API_KEY and TOGETHER_API_KEY env vars
  - Also: services/llm/groq.py (standalone) and services/llm/together.py (standalone)

  How It's Selected

  services/llm/__init__.py registers all 5 backends via @register_llm():
  - ollama — local Qwen3-30B-A3B (your current default via config.llm.default_model)
  - cloud — Groq primary + Together fallback
  - groq — Groq standalone
  - together — Together standalone
  - llama-cpp — local GGUF
  - transformers-flash — local HuggingFace

  The active backend is set by ATLAS_LLM_DEFAULT_MODEL env var. Currently "ollama".

  What You Need To Use Cloud

  1. Set GROQ_API_KEY and/or TOGETHER_API_KEY in .env
  2. Either:
    - Switch default: ATLAS_LLM_DEFAULT_MODEL=cloud
    - Or add a routing layer that selects backend per-query (does NOT exist yet)

  What's Missing: Per-Query Routing

  Currently there's no complexity-based routing. All queries go to whichever single backend is
   active. To use Qwen3 for easy stuff and cloud for hard stuff, you'd need a router that:
  1. Classifies query complexity (the intent router already partially does this)
  2. Selects LLM backend accordingly
  3. Falls back to cloud on local failure

  ---
  5. Qwen3-30B vs Cloud — Task Difficulty Map

  Qwen3-30B Handles Well (keep local)

  ┌──────────────────────────────────────┬────────────────────────────────────────┐
  │                 Task                 │             Why It's Easy              │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Intent classification (LLM fallback) │ Short JSON output, constrained         │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Device commands ("turn on X")        │ 1-step intent parse + execute          │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Parameterless tools (time, weather)  │ No LLM needed at all                   │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Reminder set/list                    │ Simple slot filling                    │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Conversation (short Q&A)             │ 1-2 sentence responses, 100 max tokens │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Security queries                     │ Tool routing, template responses       │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Presence/scene commands              │ Direct mapping                         │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Welcome home briefing                │ Template prompt, structured data       │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Departure check                      │ Builtin, no LLM                        │
  ├──────────────────────────────────────┼────────────────────────────────────────┤
  │ Morning briefing                     │ Template with data injection           │
  └──────────────────────────────────────┴────────────────────────────────────────┘

  Needs Cloud (70B+ model)

  Task: Email composition
  Why It's Hard: Must write coherent multi-paragraph body, understand tone
  ────────────────────────────────────────
  Task: Receptionist graph
  Why It's Hard: Multi-tool judgment: greet, identify, route, book — contextual decisions
  ────────────────────────────────────────
  Task: Complex booking
  Why It's Hard: 4+ step reasoning with branching (reschedule, conflict resolution)
  ────────────────────────────────────────
  Task: Calendar with natural dates
  Why It's Hard: "next Tuesday after my dentist" requires temporal reasoning
  ────────────────────────────────────────
  Task: Proactive action extraction
  Why It's Hard: Nuanced NLU on conversational text
  ────────────────────────────────────────
  Task: Long conversation context
  Why It's Hard: 6+ turn history with follow-ups
  ────────────────────────────────────────
  Task: Code gen / complex reasoning
  Why It's Hard: Anything requiring 4+ step chains

  Gray Zone (Qwen3 works but cloud is better)

  ┌───────────────────────────────┬─────────────────────────────────────────┐
  │             Task              │                  Notes                  │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ Email subject line            │ Qwen3 can do it, cloud is more polished │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ Booking confirmation phrasing │ Qwen3 is functional, cloud is natural   │
  ├───────────────────────────────┼─────────────────────────────────────────┤
  │ Error recovery / rephrasing   │ Qwen3 sometimes loops                   │
  └───────────────────────────────┴─────────────────────────────────────────┘

  ---
  6. Use Cases by Tier

  Atlas Brain (your desktop PC)

  Currently working:
  - Voice-to-voice pipeline (wake word -> STT -> LLM -> TTS)
  - LangGraph agent with 10 sub-graphs
  - 30+ registered tools
  - HA device control (lights, switches, media, climate)
  - Autonomous scheduler (6 builtin tasks + hook tasks)
  - Presence tracking with arrival/departure hooks
  - Memory (Graphiti RAG + PostgreSQL conversation store)
  - Alert system (rules engine, ntfy push, TTS delivery)
  - Identity sync (face/gait/speaker embeddings Brain <-> Edge)

  Planned/Incomplete:
  - Per-query LLM routing (local vs cloud) — not built
  - Phone V2 STT/TTS re-integration with V3 voice pipeline — pending

  Recently Completed:
  - cancel_appointment / reschedule_appointment — registered, routed, integrated into booking workflow
  - comms/ module — Active (PersonaPlex speech-to-speech, SignalWire webhooks, Resend email)
  - Calendar write tool — CreateCalendarEventTool registered, routed (calendar_write -> create_calendar_event), workflow functional
  - Gmail digest — OAuth token store integrated, auto-persist rotation, unified setup script (scripts/setup_google_oauth.py)
  - Google OAuth token management — GoogleTokenStore with file persistence (data/google_tokens.json), .env fallback, health endpoint status
  - Gmail send — GmailTransport in tools/gmail.py, EmailTool prefers Gmail when configured, Resend as fallback

  Edge Node (Orange Pi 5 Plus)

  Currently working:
  - Vision: YOLO World (94ms NPU) + motion gating
  - Face recognition: RetinaFace + MobileFaceNet (NPU)
  - Gait recognition: YOLOv8n-pose (NPU)
  - Person tracking with identity fusion
  - STT: SenseVoice int8 ONNX via sherpa-onnx (Orange Pi deployment at /opt/atlas-node/)
         Parakeet-TDT-0.6b (atlas_edge/ in repo, for GPU-equipped nodes)
  - TTS: Piper (en_US-amy-low, ~6.6x realtime on RK3588)
  - Speaker ID: Resemblyzer (256-dim embeddings, brain-side)
  - Local LLM fallback: None (static text fallback when brain unreachable)
  - WebSocket bidirectional to Brain
  - RTSP camera via MediaMTX + FFmpeg
  - Identity sync with Brain

  User-Facing Offerings (what Atlas could offer to others)

  Ready now (single-user):
  - Voice-controlled smart home (any HA setup)
  - Security monitoring with person detection + identity
  - Scheduled briefings (morning, security, device health)
  - Reminder system with voice set/complete
  - Weather/time/traffic/calendar queries
  - Push notifications (ntfy)

  Achievable with cloud routing:
  - Email composition via voice
  - Appointment booking with natural language
  - Calendar management ("schedule lunch with Maria next Thursday")
  - Multi-room presence-aware automation
  - Receptionist mode for business use

  Needs development for multi-user:
  - Per-user profiles and preferences
  - Multi-node edge deployment (multiple rooms/locations)
  - Role-based access (admin vs family vs guest)
  - Cloud API as a service (expose Brain API externally)
  - Mobile app / web dashboard
  - Multi-tenant scheduling/booking

  ---
  7. Memory System Audit (Feb 2026)

  Architecture: Two tiers, three data stores.

  Short-term (in-memory): ContextAggregator - people, objects, devices, audio events, 20-turn buffer
  Short-term (persistent): PostgreSQL conversation_turns - per-session turn history
  Long-term (knowledge):   GraphRAG (atlas-memory HTTP) - facts, preferences, personal info

  WORKING:

  - PostgreSQL turn persistence (graph path):  atlas_brain/agents/graphs/atlas.py _store_turn()
  - PostgreSQL turn persistence (streaming):   atlas_brain/voice/launcher.py _persist_streaming_turns()
  - Session creation (voice pipeline):         atlas_brain/voice/pipeline.py _ensure_session()
  - Session API (REST):                        atlas_brain/api/session.py
  - Conversation history loaded into LLM:      atlas_brain/agents/graphs/atlas.py _generate_llm_response()
  - GraphRAG READ (context retrieval):         atlas_brain/agents/graphs/atlas.py retrieve_memory node
  - Query classification (skip RAG for cmds):  atlas_brain/memory/query_classifier.py
  - Entity tracker (pronoun resolution):       atlas_brain/agents/entity_tracker.py (in-memory)

  GAPS:

  GAP-1: History query includes device commands in LLM context
    Severity: Medium
    Location: atlas_brain/agents/graphs/atlas.py line 820, atlas_brain/voice/launcher.py line 231
    Issue: Raw SQL fetches ALL turn types. Device command logs pollute LLM conversation context.
    Fix: Add AND turn_type = 'conversation' to both raw SQL queries.
    Status: CLOSED
    Resolution: Filter applied to both SQL locations. 3 new tests added (TestLLMContextQuery).
                78 tests passing (75 existing + 3 new).

  GAP-1 FIX PLAN
  ~~~~~~~~~~~~~~
  Problem: Two raw SQL queries fetch conversation history for LLM context without
  filtering by turn_type. Device commands ("Turn on lights" / "Done.") consume
  context window slots meant for actual conversation, degrading follow-up quality.

  Affected code (2 locations, identical query):

    Location A: atlas_brain/agents/graphs/atlas.py _generate_llm_response()
      Line ~820: SELECT role, content FROM conversation_turns
                 WHERE session_id = $1 ORDER BY created_at DESC LIMIT 6

    Location B: atlas_brain/voice/launcher.py _stream_llm_response()
      Line ~231: Same query

  Change: Add turn_type filter to both:
      SELECT role, content FROM conversation_turns
      WHERE session_id = $1 AND turn_type = 'conversation'
      ORDER BY created_at DESC LIMIT 6

  Why this is safe:
    - turn_type column added in migration 006_daily_sessions.sql
    - DEFAULT 'conversation' on the column, so legacy rows are included
    - Index idx_conversation_turns_type on (session_id, turn_type) exists in migration 006
    - Existing index idx_turns_session_created on (session_id, created_at DESC) covers ordering
    - ConversationRepository.get_history() already filters identically
    - test_llm_context_excludes_commands validates the repository-level behavior
    - No callers depend on commands appearing in LLM history

  Validation:
    1. Run existing 75 tests (27 persistence + 48 security) - must all pass
    2. Write a targeted test that inserts mixed turns, then calls the same raw SQL
       with the new filter, confirming commands are excluded
    3. Verify the index covers the new query (EXPLAIN ANALYZE)

  GAP-2: No real-time GraphRAG writes (long-term memory write path broken)
    Severity: High
    Location: atlas_brain/services/memory/client.py add_conversation_turn() - never called
              atlas_brain/memory/service.py store_conversation() - never called (MemoryService unused)
    Issue: Conversations are stored in PostgreSQL but never written to GraphRAG in real time.
           Only the nightly batch sync can populate long-term memory, but see GAP-3.
    Status: CLOSED
    Resolution: Wired fire-and-forget GraphRAG write into AtlasAgentMemory.add_turn().
                Both persist paths (graph _store_turn + streaming _persist_streaming_turns)
                now write conversation turns to GraphRAG via MemoryClient.add_conversation_turn().
                Gated by settings.memory.enabled + store_conversations. Command turns excluded.
                5 new tests added (TestGraphRAGWriteWiring). 83 tests passing.

  GAP-3: Nightly memory sync never scheduled
    Severity: High
    Location: atlas_brain/autonomous/runner.py line 37 (handler registered)
              atlas_brain/autonomous/scheduler.py line 87 (loads from DB)
    Issue: Handler exists but no scheduled_tasks DB row seeds it. Job never triggers.
    Status: CLOSED
    Fix: Added _DEFAULT_TASKS class variable and _ensure_default_tasks() to
         TaskScheduler in scheduler.py. Called from start() after _load_tasks_from_db().
         Seeds nightly_memory_sync (cron 0 3 * * *, timeout 300s) and
         cleanup_old_executions (cron 30 3 * * *, timeout 120s) idempotently via
         repo.get_by_name() guard. Also closes GAP-8.
         5 new tests in TestDefaultTaskSeeding (test_default_task_seeding.py).
         88 integration tests passing.

  GAP-4: In-memory context never fed to agent graph
    Severity: Low
    Location: atlas_brain/agents/memory.py (build_context_string, get_in_memory_conversation)
              atlas_brain/agents/graphs/atlas.py _generate_llm_response()
    Issue: ContextAggregator tracks people, objects, devices, audio events.
           No graph node reads runtime_context or calls build_context_string().
           Physical awareness data never reaches the LLM system prompt.
    Status: CLOSED
    Fix: Reader side -- wired ContextAggregator.build_context_string() into LLM paths:
         - atlas_brain/agents/graphs/atlas.py: awareness injected into system_parts
         - atlas_brain/voice/launcher.py: same injection plus speaker_name
         Writer side -- wired all 4 existing data sources into ContextAggregator:
         - atlas_brain/vision/subscriber.py: _update_context() feeds update_person()
           for person tracks and update_object() for all other YOLO classes
         - atlas_brain/capabilities/backends/homeassistant_ws.py: _update_context_device()
           feeds update_device() on every HA state_changed event
         - atlas_brain/voice/pipeline.py: speaker identification feeds update_person()
           with name, confidence, and node_id location
         - atlas_brain/voice/pipeline.py: start() calls set_room(node_id) at boot
         Note: Audio classification (add_audio_event) has no source yet -- YAMNet
         integration is not implemented. When added, wire into add_audio_event().
         11 tests in test_context_awareness.py + 17 tests in test_context_writers.py.

  GAP-5: MemoryService fully implemented but never wired
    Severity: Informational
    Location: atlas_brain/memory/service.py (486 lines)
    Issue: Unified layer aggregating PostgreSQL + GraphRAG + profiles + token budgets.
           get_memory_service() exported but called by zero modules.
           Does everything the graph does manually plus everything it is missing.
    Status: CLOSED
    Fix: Wired get_memory_service().gather_context() into both LLM response paths:
         - atlas_brain/agents/graphs/atlas.py: replaced raw SQL history fetch with
           MemoryService.gather_context(). Added user profile injection (name,
           response_style, expertise_level) to system prompt.
         - atlas_brain/voice/launcher.py: same gather_context() replacement + profile
           injection. Also replaced _persist_streaming_turns to use
           MemoryService.store_conversation() for dual-write (PostgreSQL + GraphRAG).
         Raw SQL for conversation_turns removed from both files.
         27 new tests in test_memory_service_wiring.py (3 singleton + 10 prompt
         formatting + 4 atlas wiring + 5 launcher wiring + 5 integration).

  GAP-6: User profiles never loaded
    Severity: Low
    Location: atlas_brain/storage/repositories/profile.py (301 lines + migration 004)
    Issue: ProfileRepository exists. MemoryService._load_user_profile() calls it.
           Since MemoryService is unused, user preferences never reach the LLM.
    Status: CLOSED (via GAP-5 fix)
    Fix: Wiring MemoryService (GAP-5) activates the _load_user_profile() code path
         inside gather_context(). When user_id is provided, ProfileRepository loads
         display_name, timezone, response_style, expertise_level, and enable_rag
         preferences. These flow into the system prompt via the new profile injection
         code in both atlas.py and launcher.py. User-to-speaker mapping is a separate
         feature -- profiles activate automatically when user_id is provided.

  GAP-7: Non-UUID session IDs silently break history + persistence
    Severity: Medium
    Location: atlas_brain/agents/graphs/atlas.py _generate_llm_response() line ~820
              atlas_brain/voice/launcher.py _stream_llm_response() line ~231
              atlas_brain/agents/memory.py add_turn() line ~148 (UUID() cast)
              atlas_brain/api/openai_compat.py (sha256 truncated to 16-char hex)
              atlas_brain/api/ollama_compat.py (same sha256 pattern)
              atlas_brain/api/edge_routes.py (edge WebSocket sends arbitrary string)
    Issue: The conversation_turns.session_id column is UUID. The voice pipeline
           produces valid UUIDs via uuid4(). But three other entry points produce
           non-UUID strings:
             - OpenAI compat API: sha256[:16] hex hash of model name
             - Ollama compat API: same sha256 pattern
             - Edge WebSocket transcript handler: arbitrary client string
           When these reach the raw SQL (WHERE session_id = $1), asyncpg raises
           InvalidTextRepresentationError. The except block silently swallows it,
           so: (a) no conversation history is loaded, (b) add_turn() fails at
           UUID(session_id), so turns are never persisted. Multi-turn context is
           silently broken for all non-voice entry points.
    Status: CLOSED
    Fix: Created atlas_brain/utils/session_id.py with normalize_session_id()
         (UUID5 deterministic mapping for non-UUID strings) and ensure_session_row()
         (idempotent INSERT INTO sessions). Wired into all 5 entry points:
           - atlas_brain/api/openai_compat.py (replaced sha256[:16] with normalize)
           - atlas_brain/api/ollama_compat.py (same)
           - atlas_brain/api/llm.py (normalize arbitrary request.session_id)
           - atlas_brain/api/comms/webhooks.py (normalize telephony call_id)
         Hardened downstream consumers:
           - atlas_brain/agents/graphs/atlas.py: cast session_id to UUID before asyncpg
           - atlas_brain/voice/launcher.py: same
           - atlas_brain/agents/memory.py: normalize before UUID() cast in get/add
         14 new tests in test_session_id_normalization.py (9 unit + 5 integration).
         45 integration tests passing.

  GAP-8: No PostgreSQL cleanup mechanism is functional (unbounded table growth)
    Severity: High
    Location: atlas_brain/jobs/nightly_memory_sync.py _purge_old_messages()
              atlas_brain/autonomous/runner.py (cleanup_job handler also registered)
    Issue: Two cleanup mechanisms exist but neither runs:
             - nightly_memory_sync purges conversation_turns > N days (GAP-3 blocks it)
             - cleanup_job purges task_executions, presence_events, proactive_actions
           Both are registered as handlers but have no scheduled_tasks DB row.
           No migration seeds any scheduled task rows. Tables grow without bound.
    Status: CLOSED (via GAP-3 fix -- both tasks now auto-seeded on scheduler start)

  GAP-9: Voice pipeline bypasses SessionRepository, fragments conversation history
    Severity: Low
    Location: atlas_brain/voice/pipeline.py (session creation via uuid4())
    Issue: The voice pipeline creates sessions via direct uuid4() + raw SQL INSERT
           rather than using SessionRepository.get_or_create_session(). This means:
             - Every voice pipeline restart creates a new session UUID
             - Daily session reuse logic in SessionRepository is bypassed
             - Conversation history is fragmented across sessions on restart
    Status: CLOSED
    Fix: Replaced raw SQL INSERT in VoicePipeline._ensure_session() with
         SessionRepository. New logic:
         1. Queries for today's active session matching terminal_id (node_id)
         2. If found, reuses that session_id and calls touch_session()
         3. If not found, creates via repo.create_session(terminal_id=node_id)
         This gives daily session continuity per voice node. Conversation history
         now persists across pipeline restarts on the same day.
         10 new tests in test_voice_session_reuse.py (5 wiring + 5 integration).
