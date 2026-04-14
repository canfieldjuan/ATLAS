# Atlas Architectural Manifest

## Core Philosophy: "Deterministic Infrastructure for Non-Deterministic Intelligence"
The Atlas system is engineered as a tiered, event-driven ecosystem that separates high-precision, low-latency conversational interaction from high-volume, high-accuracy asynchronous reasoning. It treats LLMs as powerful but non-deterministic "instruments" that must be governed by rigid, deterministic "scorecards" and "contracts."

---

## 1. The "Atlas Brain" (The Conversational Conductor)
*   **Role:** Real-time orchestrator for voice and text interaction.
*   **Key Components:**
    *   **Voice Pipeline:** Low-latency STT (Nemotron) and TTS (Kokoro) using Pipecat.
    *   **LangGraph Orchestration:** A modular "Modes" architecture (Home, Scheduling, Security) that routes intent based on semantic centroids.
    *   **Hardware Determinism:** Global `cuda_lock` for serializing GPU access, ensuring stable performance on fixed VRAM budgets.
*   **Methodology:** Tiered inference (Local vs. Cloud) to balance cost, latency, and reasoning depth.

---

## 2. The "Churn Signals" (The Industrial Data Refinery)
*   **Role:** High-throughput B2B market intelligence engine.
*   **Key Components:**
    *   **Data Governance:** A strict "Field Ownership Contract" (`_b2b_field_contracts.py`) that regulates every piece of enrichment data.
    *   **Stratified Reasoning:** Separates deterministic data aggregation (Weekly Pools) from asynchronous LLM synthesis (Narratives/Briefings).
    *   **The Witness System:** A fact-checking layer that verifies evidence spans and salience before reporting.
*   **Methodology:** Industrial-scale batch processing (Anthropic/OpenRouter) with automated content generation (Blogs, Battle Cards).

---

## 3. "Mission Control" (The Surface Layers)
*   **Role:** Real-time observability and business intelligence dashboards.
*   **Churn UI:** Transforms raw scraping and signals into actionable market archetypes (Pricing Shock, Feature Gap) and displacement flows.
*   **Admin Cost UI:** A high-resolution telemetry suite for the "AI Economy."
    *   **Unit Economics:** Real-time tracking of cost-per-call, token usage, and provider spend.
    *   **System Health:** Monitoring GPU/VRAM pressure, scraping success rates, and Reddit signal conversion funnels.
*   **Methodology:** Every "chemical reaction" in the backend refinery is surfaced to the UI, providing a "God Mode" view of the entire intelligence pipeline.

---

## 4. Strategic Revenue Engines (Algorithmic Prospecting)
*   **Role:** Autonomous lead-to-revenue conversion pipeline.
*   **Key Components:**
    *   **Semantic Pain-Matching Layer:** Cross-references raw competitor review data (negative sentiment) with normalized company profiles and decision-maker archetypes (Founders/SMBs).
    *   **Dynamic Content Mapping:** Automatically selects or generates targeted blog posts that address the specific "Pain Point" identified in the lead signal.
    *   **Affiliate Intelligence Integration:** Injects context-aware revenue opportunities (affiliate links) into generated content with 100% link accuracy.
    *   **Trigger-Based Outreach:** Initiates personalized email campaigns (via Smartlead/Instantly) that synchronize the "Pain Narrative" from the initial review to the final solution blog.
*   **Methodology:** Prioritizing **"TCV Velocity"** (Total Contract Value speed) by targeting SMBs and Solo-Founders to bypass enterprise procurement friction.

---

## 5. The Coding Methodology: "The Symphony & The Refinery"
The Atlas methodology avoids "AI Slop" by enforcing strict canonical standards and engineering modularity.
*   **The Symphony:** Orchestrating multiple models and tools into a single, cohesive user experience.
*   **The Refinery:** Scaling raw data into refined intelligence through governed, automated pipelines.

---

## 6. Skill Set Mapping: The "Symphony Director"
| Role | Skill | Evidence in Atlas |
| :--- | :--- | :--- |
| **System Architect** | Principal-level design | Designing the tiered LangGraph Modes and the Churn Refinery pools. |
| **Operations Engineer** | Resource & Cost management | Building the Admin Cost UI and managing `cuda_lock` hardware contention. |
| **Data Governance Lead** | Structural Integrity | Enforcing the B2B Field Contracts and automated governance tests. |
| **Full-Stack Orchestrator** | End-to-End Delivery | Connecting deep-reasoning Python backends to high-fidelity React dashboards. |
| **Market Strategist** | Domain Intelligence | Translating raw scraping into Churn Archetypes and competitive displacement signals. |

---

*Authored by: The System Director*
*Date: April 2026*
