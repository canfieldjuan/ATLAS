
https://accounts.google.com/o/oauth2/v2/auth?client_id=875644967642-fa81kcprqhe501qsa249vc0aacrocbjp.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A8085&response_type=code&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fgmail.readonly&access_type=offline&prompt=consent



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
  Status: Partial
  Purpose: Phone/PersonaPlex processors, tool_bridge. Appears partially wired
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
  Purpose: LLM backends (5), embedding, intent router, speaker ID, VLM, tool executor,
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
  ├────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ cancel_appointment     │ scheduling.py │ Not registered in __init__.py       │
  ├────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ reschedule_appointment │ scheduling.py │ Not registered in __init__.py       │
  └────────────────────────┴───────────────┴─────────────────────────────────────┘

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
  - comms/ module (PersonaPlex phone handling) — partially wired
  - cancel_appointment / reschedule_appointment — defined but not registered
  - Gmail tool (OAuth partially set up) — token needed
  - Calendar write tool (OAuth) — token needed

  Edge Node (Orange Pi 5 Plus)

  Currently working:
  - Vision: YOLO World (94ms NPU) + motion gating
  - Face recognition: RetinaFace + MobileFaceNet (NPU)
  - Gait recognition: YOLOv8n-pose (NPU)
  - Person tracking with identity fusion
  - STT: SenseVoice (16x realtime CPU)
  - TTS: Kokoro-82M (0.38x realtime CPU)
  - Speaker ID: CampPlus (192-dim embeddings)
  - Local LLM fallback: Phi-3-mini Q4 (slow but functional)
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
