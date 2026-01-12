# Atlas Build Log

**Planning Note:** Active plans now live in `PLAN.md`. This log captures completed sessions and key decisions only.

## Session 1: 2026-01-07 - Architectural Design

**Objective:** Define the high-level architecture for the Atlas project.

**Decisions:**

1.  **Initial Architecture (1.0):** Proposed a modular, multi-model system with separate components for Wake Word, STT, CV, NLU (LLM), and TTS.
2.  **Refined Architecture (2.0):** Updated the design to use a central Vision-Language Model (VLM) to merge the NLU and CV components, simplifying the core logic while retaining a specialized audio pipeline.
3.  **Final Architecture (3.0):** Evolved the concept into a client-server model. Atlas will be a centralized "Brain" running on a server with a GPU, exposing its services via an API. Other devices ("Terminals") will connect to this brain over the network. This is the guiding design principle.

**Outcome:** The project will be developed as a centralized "intelligence platform" (The Atlas Brain) that serves multiple lightweight "terminals".


## Session 2: 2026-01-07 - Environment Scaffolding

**Objective:** Create the basic project structure and Python environment for the server.

**Actions:**

1.  **Directory Structure:** Created the core project directories (`atlas_brain`, `logs`, `models`, `scripts`).
2.  **Dependencies:** Established a `requirements.txt` file with `fastapi`, `uvicorn`, and `python-dotenv`.
3.  **Isolation:** Set up a `.gitignore` file and created a local Python virtual environment (`.venv`).
4.  **Installation:** Installed the specified dependencies into the virtual environment.

**Outcome:** The project now has a clean, isolated environment and the necessary foundational folder structure.


## Session 3: 2026-01-07 - "Hello World" API

**Objective:** Create and verify a basic API endpoint to confirm the server setup.

**Actions:**

1.  **API Code:** Created `atlas_brain/main.py` with a simple FastAPI app and a `/ping` endpoint.
2.  **Debugging:** Encountered significant environment-specific issues when attempting to run the `uvicorn` server as a background process.
    *   Initial attempts failed due to `[Errno 98] Address already in use`, as port 8000 was occupied.
    *   After clearing the port, the server continued to fail silently when launched in the background, with no logs being captured.
    *   **Conclusion:** Running foreground processes is reliable, but background processes (`&`) via the tool environment are unstable for this specific server.
3.  **Resolution:** The user started the server manually in their own terminal.
4.  **Verification:** Successfully connected to the server's `/ping` endpoint and received the expected `{"status":"ok","message":"pong"}` response.

**Outcome:** The basic FastAPI server is functional. The environmental issues with background processes are noted, and will be permanently solved by dockerizing the application.


## Session 4: 2026-01-07 - Dockerization

**Objective:** Containerize the FastAPI server to ensure a stable and reproducible environment.

**Actions:**

1.  **Configuration:** Created a `.dockerignore` file, a `Dockerfile` defining the server image, and a `docker-compose.yml` file to manage the container.
2.  **Debugging `docker-compose`:** The initial `docker compose up` command failed to expose the server's port. The `docker ps` command showed an empty `PORTS` column.
3.  **Workaround:** Proved that the Docker engine itself was working by using `docker run -p 8000:8000`, which successfully started the container and mapped the port.
4.  **Resolution:** Identified and removed the obsolete `version: '3.8'` tag from `docker-compose.yml`. This resolved the issue.
5.  **Verification:** Successfully started the server using `docker compose up -d`. The `curl` command to the `/ping` endpoint returned the expected success response.

**Outcome:** The Atlas Brain server is now fully containerized and running reliably via `docker-compose`. Phase 1 is complete.


## Session 5: 2026-01-07 - API Refactoring

**Objective:** Refactor the API into a more scalable, modular structure.

**Actions:**

1.  **APIRouter:** Created a new file `atlas_brain/api/query.py` to house an `APIRouter`.
2.  **Modularization:** Moved the `/ping` health check endpoint into the new router.
3.  **Integration:** Modified `atlas_brain/main.py` to import the new router and include it with a `/api/v1` prefix.
4.  **Verification:** Restarted the Docker container and confirmed the endpoint was successfully moved to `http://127.0.0.1:8000/api/v1/ping`.

**Outcome:** The API now has a scalable structure, allowing us to organize endpoints into logical modules.


## Session 6: 2026-01-07 - Text Query Endpoint

**Objective:** Implement the first core API endpoint for handling simple text-based queries.

**Actions:**

1.  **Schema:** Created `atlas_brain/schemas/query.py` to define a `TextQueryRequest` Pydantic model for the request body.
2.  **Endpoint:** Added a `POST /api/v1/query/text` endpoint to the `query.py` router.
3.  **Implementation:** The endpoint validates the incoming request using the Pydantic model and returns a mocked JSON response.
4.  **Verification:** Restarted the container and successfully tested the new endpoint with `curl`, receiving the expected mocked response.

**Outcome:** The Atlas Brain can now accept and process basic text queries via a structured API endpoint.


## Session 7: 2026-01-07 - Audio Query Endpoint

**Objective:** Implement an API endpoint for handling audio file uploads.

**Actions:**

1.  **Dependency:** Added `python-multipart` to `requirements.txt` to support file uploads.
2.  **Docker Image:** Rebuilt the Docker image to include the new dependency.
3.  **Endpoint:** Added a `POST /api/v1/query/audio` endpoint to `query.py`.
4.  **Implementation:** The endpoint uses FastAPI's `UploadFile` to accept a file and returns a mocked JSON response with the file's metadata.
5.  **Verification:** Created a dummy `.wav` file and successfully tested the upload functionality using `curl -F`.

**Outcome:** The Atlas Brain can now accept audio files via a structured API endpoint.


## Session 8: 2026-01-07 - Vision Query Endpoint

**Objective:** Implement an API endpoint for handling image file uploads with an optional text prompt.

**Actions:**

1.  **Endpoint:** Added a `POST /api/v1/query/vision` endpoint to `query.py`.
2.  **Implementation:** The endpoint uses FastAPI's `UploadFile` and `Form` to accept both a file and an optional text field in a single multipart/form-data request.
3.  **Verification:** Created a dummy `.jpg` file and successfully tested the upload functionality using `curl` with multiple `-F` flags, receiving the expected mocked response.

**Outcome:** The Atlas Brain can now accept image files and associated text prompts via a structured API endpoint.


## Session 9: 2026-01-07 - Service Layer Refactoring

**Objective:** Decouple the API layer from the business logic by creating a service layer.

**Actions:**

1.  **Service Placeholders:** Created `atlas_brain/services/vlm.py` and `atlas_brain/services/stt.py` to house placeholder functions for future AI model logic.
2.  **Refactoring:** Modified the API endpoints in `atlas_brain/api/query.py` to call their corresponding service functions instead of containing logic themselves.
3.  **Verification:** Restarted the container and tested the `/api/v1/query/text` endpoint. The response from the placeholder service was returned successfully, confirming the refactoring was successful.

**Outcome:** The project now has a clean separation between the API/routing layer and the business logic/service layer. This makes the architecture much more maintainable and scalable. Phase 2 is complete.


## Phase 3: AI Model Integration (LLM)

**Objective:** Replace the mocked text query service with a live, running Large Language Model.

**Actions:**

1.  **Initial Plan:** The initial approach was to use the `microsoft/Phi-3-mini-4k-instruct` model and a GPU-enabled Docker image.
2.  **Environment Failure:** Hit a critical error (`could not select device driver "nvidia"`) due to the host environment missing the **NVIDIA Container Toolkit**.
3.  **Debugging Host Environment:** Guided the user through multiple attempts to install the toolkit, facing several copy-paste and shell interpretation errors. The issue was resolved by providing a shell script (`install_nvidia_toolkit.sh`) that the user could execute, which successfully configured the system.
4.  **Model Loading Failure:** After enabling the GPU, the initial attempt to load the Phi-3 model resulted in `Connection reset by peer` errors, indicating a crash during inference, likely due to a GPU VRAM OOM (Out Of Memory) error.
5.  **Pivot to Moondream2:** Based on user feedback and the instability of the large model, the strategy was changed to use the much smaller and more efficient `vikhyatk/moondream2` model.
6.  **Dependency Debugging:** The switch to `moondream2` revealed a missing dependency, `Pillow`, which was added to `requirements.txt`.
7.  **Code Debugging:** After fixing the dependency, testing revealed `TypeError` and `ValueError` exceptions in the service layer code due to an incorrect understanding of the `moondream2` API for text-only queries.
8.  **Final Fix:** The code was corrected to provide a blank, dummy `PIL.Image` object for text-only queries, satisfying the model's API.
9.  **Success!** A final test of the `POST /api/v1/query/text` endpoint returned a live, generated response from the `moondream2` model: `"I walk."`.

**Outcome:** The Atlas Brain server can now process text queries using a live AI model. The core pipeline from API endpoint to model inference is fully functional.


## Session 10: 2026-01-10 - Conversation Persistence Planning

**Objective:** Design and plan a persistent conversation storage system to enable seamless multi-location conversations (office → car → home without context loss).

**Analysis Performed:**

1.  **Codebase Audit:** Thoroughly analyzed the existing Atlas codebase to understand current conversation handling:
    - `orchestration/context.py`: In-memory `ContextAggregator` with `ConversationTurn` dataclass
    - `api/llm.py`: Chat endpoint with no session/user tracking
    - `config.py`: Pydantic Settings pattern for configuration
    - No existing database layer

2.  **Integration Points Identified:**
    - `context.py:278-306` - `add_conversation_turn()` and `get_conversation_history()`
    - `api/llm.py:40-44` - `ChatRequest` model needs session_id
    - `api/llm.py:114-135` - `/chat` endpoint needs history loading
    - `main.py:24-89` - Lifespan handler needs DB init/shutdown
    - `config.py:100-131` - Needs `DatabaseConfig` class

3.  **Database Selection Analysis:**
    - SQLite: ~0.1-1ms latency, single-node only
    - PostgreSQL (local): ~1-5ms latency, multi-node ready
    - Self-hosted Supabase: ~5-20ms latency, full platform
    - **Decision:** PostgreSQL with `asyncpg` driver for lowest latency + scalability

**Deliverables:**

1.  Created `CONVERSATION_PERSISTENCE_PLAN.md` with:
    - 5-phase implementation plan
    - Database schema design
    - Exact code modification points
    - Validation criteria per phase
    - Rollback strategy
    - Latency targets (<2ms for writes, <3ms for reads)

**Key Design Decisions:**

1.  **PostgreSQL over SQLite:** Multi-terminal support required from day one
2.  **asyncpg over SQLAlchemy:** Raw driver is faster, ORM complexity not needed
3.  **Repository pattern:** Clean separation, testable components
4.  **Backward compatibility:** Existing endpoints work without session_id
5.  **Feature flags:** Database can be disabled for fallback to in-memory

**Outcome:** Comprehensive implementation plan created and awaiting user approval. No code changes made yet - plan first, then execute.


## Session 11: 2026-01-10 - Conversation Persistence Phase 1 & 2 Implementation

**Objective:** Implement the database infrastructure and repository layer for conversation persistence.

**Actions:**

1.  **PostgreSQL Setup:**
    - Added PostgreSQL service to `docker-compose.yml` with healthcheck
    - Started container: `atlas_postgres` running PostgreSQL 16-alpine
    - Added `data/postgres/` to `.gitignore`

2.  **Storage Module Created:**
    - `atlas_brain/storage/__init__.py` - Module exports
    - `atlas_brain/storage/config.py` - DatabaseConfig with pydantic-settings (env prefix: ATLAS_DB_)
    - `atlas_brain/storage/database.py` - DatabasePool with asyncpg connection pooling
    - `atlas_brain/storage/models.py` - User, Session, ConversationTurn, Terminal dataclasses
    - `atlas_brain/storage/repositories/__init__.py` - Repository exports
    - `atlas_brain/storage/repositories/conversation.py` - ConversationRepository (add_turn, get_history)
    - `atlas_brain/storage/repositories/session.py` - SessionRepository (get_or_create_session, multi-terminal)
    - `atlas_brain/storage/migrations/__init__.py` - Migration runner
    - `atlas_brain/storage/migrations/001_initial_schema.sql` - Initial database schema

3.  **Database Schema Created:**
    - Tables: `users`, `sessions`, `conversation_turns`, `terminals`, `schema_migrations`
    - Indexes optimized for low-latency queries
    - UUID primary keys with `uuid-ossp` extension

4.  **Integration with FastAPI:**
    - Modified `main.py` to import storage module
    - Added database initialization in lifespan startup handler
    - Added database cleanup in lifespan shutdown handler

5.  **Environment Configuration:**
    - Added database environment variables to `.env`
    - Added `asyncpg>=0.29.0` to `requirements.txt`
    - Installed `pydantic`, `pydantic-settings`, `asyncpg` in venv

**Verification:**

```
✅ PostgreSQL container running (atlas_postgres)
✅ Connection pool initializes successfully
✅ Tables created via migration script
✅ Session creation works
✅ Conversation turn creation/retrieval works
✅ Multi-terminal session continuity works
✅ No import errors or syntax errors
```

**Next Steps:** Phase 3 - Integration with existing conversation flow (context.py, api/llm.py)


## Session 12: 2026-01-10 - Conversation Persistence Phase 3 Implementation

**Objective:** Integrate the storage layer with the existing API layer for conversation persistence.

**Actions:**

1.  **Updated `api/llm.py`:**
    - Added imports: `logging`, `UUID`, storage module
    - Added `session_id`, `user_id`, `terminal_id` fields to `ChatRequest` model
    - Modified `/chat` endpoint to load conversation history from DB when session_id provided
    - Modified `/chat` endpoint to persist new turns (user + assistant) to DB
    - All changes backward compatible - existing API works without session_id

2.  **Created `api/session.py`:**
    - `POST /api/v1/session/create` - Create new conversation session
    - `POST /api/v1/session/continue` - Continue session from different terminal
    - `GET /api/v1/session/{session_id}` - Get session details
    - `GET /api/v1/session/{session_id}/history` - Get conversation history
    - `POST /api/v1/session/{session_id}/close` - Close/deactivate session
    - `GET /api/v1/session/status/db` - Check database connection status

3.  **Updated `api/__init__.py`:**
    - Added import for session router
    - Included session_router in main API router

**Verification:**

```
All imports work correctly
Session creation and retrieval works
Conversation persistence works
History retrieval works
ChatRequest backward compatible (session_id optional)
No breaking changes to existing API
Syntax checks pass for all modified files
```

**API Usage Example:**

```bash
# Create a session
curl -X POST http://localhost:8000/api/v1/session/create \
  -H "Content-Type: application/json" \
  -d '{"terminal_id": "office"}'

# Chat with session persistence
curl -X POST http://localhost:8000/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello Atlas"}],
    "session_id": "<session-uuid-from-above>"
  }'

# Get history
curl http://localhost:8000/api/v1/session/<session-uuid>/history
```

**Next Steps:** Phase 4 - Multi-terminal session continuity (requires user registration)


---


## Session 18: 2026-01-10 - Voice Interface & Intelligent Routing Planning

**Objective:** Fix voice input pipeline and plan intelligent model routing system.

**Actions:**

1. **Service Activation Debugging:**
   - Discovered all AI services (STT, TTS, LLM) were not auto-loading despite configuration
   - Manually activated STT (faster-whisper), TTS (piper), and LLM (Ministral-3B)
   - Verified services active via `/api/v1/health` endpoint
   - **Root Cause:** Auto-load config in main.py wasn't triggering properly (requires investigation)

2. **Audio Recording Bug Fix:**
   - Identified critical bug in `useAudioRecorder.ts` hook
   - Problem: `isRecording` state variable not captured in audio processor closure
   - **Solution:** Added `isRecordingRef` to persist recording state across closures
   - Added console logging for audio chunk debugging
   - Audio now properly captured and streamed to WebSocket

3. **Voice Pipeline Verification:**
   - Confirmed WebSocket connection established (ws://localhost:8000/api/v1/ws/orchestrated)
   - Verified audio recording: 16kHz mono PCM16 format
   - Tested orchestrator with text command (works correctly)
   - **Issue:** Audio stream not reaching orchestrator (under investigation)

4. **Model Routing Analysis:**
   - Analyzed current architecture: static single-model approach (Hermes-8B)
   - Identified available models:
     * Simple: Ministral-3B (1.9GB) - Currently active
     * Medium: Hermes-8B (4.6GB)
     * Complex: BlackSheep-24B (14GB), Xortron-24B (14GB)
   - **Finding:** No dynamic routing exists - all queries use same model

5. **Comprehensive Planning:**
   - Created `INTELLIGENT_MODEL_ROUTING_PLAN.md` with 5-phase implementation
   - Verified all 17 usages of `llm_registry` for breaking change analysis
   - Designed feature-flag protected rollout strategy
   - Documented rollback procedures and performance considerations
   - **Key Decision:** Use 3-tier system (simple/medium/complex) with configurable thresholds

**Architecture Analysis:**

Current request flow:
```
Voice Input → STT → Intent Parser (VLM) → [Device Action OR LLM Response] → TTS
                                              ↑
                                         STATIC MODEL
                                         (Hermes-8B)
```

Planned intelligent routing:
```
Voice Input → STT → Intent Parser (VLM) → [Device Action OR LLM Response] → TTS
                                              ↑
                                    ┌─────────┴─────────┐
                                    │ Complexity Analyzer│
                                    │   Query Router    │
                                    └───────┬───────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        │                   │                   │
                   Simple (Fast)       Medium (Balanced)   Complex (Smart)
                   Ministral-3B        Hermes-8B           BlackSheep-24B
                   <1s response        2-3s response       5-10s response
```

**Files Created:**
- `INTELLIGENT_MODEL_ROUTING_PLAN.md` - Comprehensive implementation plan with 5 phases

**Files Modified:**
- `atlas-ui/src/hooks/useAudioRecorder.ts` - Fixed recording state bug with ref
- `atlas-ui/src/hooks/useAtlas.ts` - Added audio send logging

**Configuration Verified:**
- STT: faster-whisper (small.en) on CUDA ✅
- TTS: piper (en_US-libritts-high) on CPU ✅
- LLM: Ministral-3B on CUDA ✅ (should be Hermes-8B per .env)
- VLM: moondream on CUDA ✅

**Technical Findings:**

1. **Service Registry Architecture:**
   - Thread-safe with Lock for hot-swapping
   - Supports factories and direct class registration
   - Unloads previous model before loading new one
   - Compatible with planned routing system ✅

2. **Breaking Change Analysis:**
   - All 17 llm_registry usages verified safe
   - PipelineContext extensible with optional fields
   - Config system supports nested ROUTING__ prefix
   - No protocol changes needed ✅

3. **Performance Considerations:**
   - Model swap overhead: 5-10 seconds (one-time cost)
   - VRAM available: 24GB (sufficient for all tiers)
   - Routing decision overhead: <5ms (negligible)

**Outstanding Issues:**

1. **Auto-Load Failure:**
   - LLM config shows `ATLAS_LOAD_LLM_ON_STARTUP=true`
   - Services required manual activation via API
   - **Action:** Need to investigate main.py startup sequence

2. **Audio Pipeline:**
   - Audio recording works, WebSocket connected
   - Audio chunks not triggering transcription
   - **Hypothesis:** Need to verify orchestrator audio stream processing

3. **Model Mismatch:**
   - .env specifies Hermes-8B
   - Running Ministral-3B (manually activated)
   - **Action:** Restart with proper auto-load to verify config

**Implementation Plan Summary:**

| Phase | Description | Breaking? | Time Est. |
|-------|-------------|-----------|-----------|
| 1 | Query Complexity Analyzer | No | 2-3h |
| 2 | Model Routing Configuration | No | 1h |
| 3 | Model Router Service | No | 3-4h |
| 4 | Orchestrator Integration | Yes (flagged) | 4-6h |
| 5 | API Monitoring & Telemetry | No | 1-2h |

**Outcome:** Comprehensive plan created for intelligent model routing. All dependencies verified safe. Feature-flag strategy ensures no breaking changes. Ready for phased implementation when approved.

**Next Session Goals:**
1. Fix audio pipeline (verify orchestrator receives stream)
2. Investigate and fix auto-load issue  
3. Begin Phase 1 implementation (Complexity Analyzer) if approved
