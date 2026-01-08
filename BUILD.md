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
