# Mode Agents Design - Discussion Log

**Date:** 2026-01-19
**Status:** Design Finalized, Ready for Implementation

---

## Overview

AtlasAgent acts as a router that delegates to mode-specific agents. Each mode agent is responsible for its own tools and logic.

---

## Architecture

```
                        User Input
                             │
                             ▼
                    ┌─────────────────┐
                    │   AtlasAgent    │
                    │    (Router)     │
                    │                 │
                    │ • Mode tracking │
                    │ • Mode switch   │
                    │ • 2 min timeout │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   HomeAgent     │ │ ReceptionistAgent│ │  SecurityAgent  │
│   (DEFAULT)     │ │                 │ │                 │
│                 │ │                 │ │                 │
│ • Device cmds   │ │ • Booking flow  │ │ • Camera queries│
│ • Fast path     │ │ • State machine │ │ • Fast path     │
│ • No state      │ │ • Multi-turn    │ │                 │
│                 │ │                 │ │                 │
│ Tools:          │ │ Tools:          │ │ Tools:          │
│ • lights_*      │ │ • book_appt     │ │ • list_cameras  │
│ • media_*       │ │ • check_avail   │ │ • show_feed     │
│ • scene_*       │ │ • set_reminder  │ │ • get_detections│
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────┴────────┐
                    │   CommsAgent    │
                    │                 │
                    │ • Personal comms│
                    │ • Reminders     │
                    │ • Calendar      │
                    │                 │
                    │ Tools:          │
                    │ • send_email    │
                    │ • set_reminder  │
                    │ • get_calendar  │
                    └─────────────────┘
```

---

## Design Decisions

### 1. Mode Switching

**Strict phrase required:** `"Atlas switch to [mode] mode"`

Examples:
- "Atlas switch to scheduling mode"
- "Atlas switch to home mode"
- "Atlas switch to security mode"

**Aliases** (handled in ModeManager):
- "scheduling" / "appointment" / "business" → RECEPTIONIST
- "home" / "device" / "devices" → HOME
- "security" / "camera" / "cameras" → SECURITY
- "comms" / "personal" / "communications" → COMMS

### 2. Default Mode

**HOME** is the default mode.
- Most common use case
- Fast path, no state machine overhead
- Falls back here on timeout

### 3. Timeout Behavior

**2 minutes of inactivity** triggers fallback to HOME mode.

- Timer starts after last command is **completed** by model
- Timer resets on any new interaction
- If no activity for 2 mins → switch to HOME
- Timeout does NOT interrupt active workflows (state machine in progress)

**Implementation:**
```python
# In AtlasAgent or ModeManager
last_activity_time: float = time.time()
MODE_TIMEOUT_SECONDS = 120  # 2 minutes

def check_timeout(self):
    if self.current_mode == ModeType.HOME:
        return  # Already in default

    if self.workflow_state is not None:
        return  # Don't timeout during active workflow

    if time.time() - self.last_activity_time > MODE_TIMEOUT_SECONDS:
        self.switch_mode(ModeType.HOME)
        logger.info("Mode timeout - switched to HOME")
```

### 4. Mode Agents Structure

| Agent | Status | Purpose | State Machine? |
|-------|--------|---------|----------------|
| **HomeAgent** | NEW | Device commands | NO - fast path |
| **ReceptionistAgent** | EXISTS | Business booking | YES |
| **SecurityAgent** | NEW | Camera/monitoring | NO - fast path |
| **CommsAgent** | NEW | Personal comms | MAYBE (complex scheduling) |

**Each agent is self-contained:**
- Imports its own tools
- Handles its own logic
- No tool passing from AtlasAgent

### 5. Tool Ownership

**Each mode agent owns its tools directly.**

AtlasAgent does NOT filter or pass tools. Each agent imports what it needs:

```python
# HomeAgent
from ..tools import lights_near_user, media_near_user, scene_near_user

# ReceptionistAgent
from ..tools import book_appointment, check_availability, set_reminder

# SecurityAgent
from ..tools import list_cameras, show_camera_feed, get_current_detections
```

**Benefits:**
- Clear ownership
- No confusion about what tools are available
- Easier to test each agent in isolation

### 6. Shared Tools

**Not implementing shared tools yet.**

Focus on getting each mode working flawlessly first. Shared tools (time, weather, location) can be added later:
- Either in AtlasAgent (handle before routing)
- Or available in all mode agents

### 7. Workflow State Persistence

If timeout occurs during active workflow:
- **Don't timeout** - workflow in progress means user is engaged
- Only timeout when `workflow_state == None` (idle)

If user manually switches mode during workflow:
- Clear workflow state
- Start fresh in new mode

---

## Mode Configuration Reference

From `atlas_brain/modes/config.py`:

```python
ModeType.HOME: ModeConfig(
    name="home",
    tools=["lights_near_user", "media_near_user", "scene_near_user", "where_am_i"],
    model_preference="qwen3:8b",
    keywords=["turn on", "turn off", "light", "tv", "play", "pause"],
)

ModeType.RECEPTIONIST: ModeConfig(
    name="receptionist",
    tools=["check_availability", "book_appointment", "set_reminder", ...],
    model_preference="qwen3:14b",
    keywords=["appointment", "estimate", "book", "schedule"],
)

ModeType.SECURITY: ModeConfig(
    name="security",
    tools=["list_cameras", "show_camera_feed", "get_current_detections", ...],
    model_preference="qwen3:14b",
    keywords=["camera", "security", "motion", "who", "intruder"],
)

ModeType.COMMS: ModeConfig(
    name="comms",
    tools=["send_email", "set_reminder", "get_calendar"],
    model_preference="qwen3:14b",
    keywords=["call", "text", "email", "mom", "dad"],
)
```

---

## AtlasAgent Router Logic

```python
class AtlasAgent(BaseAgent):
    def __init__(self):
        self._mode_manager = get_mode_manager()
        self._mode_agents = {
            ModeType.HOME: HomeAgent(),
            ModeType.RECEPTIONIST: ReceptionistAgent(),
            ModeType.SECURITY: SecurityAgent(),
            ModeType.COMMS: CommsAgent(),
        }
        self._last_activity = time.time()

    async def run(self, context: AgentContext) -> AgentResult:
        # 1. Check timeout
        self._check_mode_timeout()

        # 2. Check for mode switch command
        mode_switch = self._mode_manager.parse_mode_switch(context.input_text)
        if mode_switch:
            self._mode_manager.switch_mode(mode_switch)
            self._last_activity = time.time()
            return AgentResult(
                success=True,
                response_text=f"Switched to {mode_switch.value} mode.",
            )

        # 3. Delegate to current mode agent
        current_mode = self._mode_manager.current_mode
        agent = self._mode_agents[current_mode]

        result = await agent.run(context)

        # 4. Update activity timestamp
        self._last_activity = time.time()

        return result
```

---

## Implementation Order

1. **Phase 1: Update AtlasAgent as Router**
   - Add mode agent delegation
   - Add timeout check (2 mins)
   - Keep existing HOME logic inline initially

2. **Phase 2: Extract HomeAgent**
   - Move device command logic from AtlasAgent
   - Fast path, no state machine

3. **Phase 3: Verify ReceptionistAgent**
   - Already exists and works
   - Ensure it integrates with router

4. **Phase 4: Create SecurityAgent**
   - Camera queries
   - Fast path

5. **Phase 5: Create CommsAgent**
   - Personal communications
   - Evaluate if state machine needed

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `atlas_brain/agents/atlas.py` | MODIFY - Add router logic, timeout |
| `atlas_brain/agents/home.py` | CREATE - Extract device commands |
| `atlas_brain/agents/security.py` | CREATE - Camera queries |
| `atlas_brain/agents/comms.py` | CREATE - Personal comms |
| `atlas_brain/agents/receptionist.py` | VERIFY - Already works |
| `atlas_brain/modes/manager.py` | MODIFY - Update timeout to 2 mins |

---

## Test Cases

### Mode Switching
```
User: "Atlas switch to scheduling mode"
→ Mode: RECEPTIONIST
→ Response: "Switched to scheduling mode."

User: [2 min inactivity]
→ Mode: HOME (automatic)
→ Log: "Mode timeout - switched to HOME"
```

### Mode-Specific Behavior
```
[Mode: HOME]
User: "Turn on the living room lights"
→ HomeAgent handles → Device command executed

[Mode: RECEPTIONIST]
User: "Book an estimate"
→ ReceptionistAgent handles → State machine starts
User: "Sarah Johnson, 789 Main St, tomorrow morning"
→ Data collected → Confirmation requested
User: "Yes"
→ book_appointment executed → Calendar event created
```

### Timeout During Workflow
```
[Mode: RECEPTIONIST, workflow: booking, phase: COLLECTING]
User: [2 min inactivity]
→ NO TIMEOUT (workflow in progress)
→ Stays in RECEPTIONIST mode
→ Workflow state preserved
```

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| Generic vs specific state machines? | Specific per agent (ReceptionistAgent pattern) |
| Shared tools? | Not yet - focus on modes first |
| Tool filtering? | Each agent owns its tools |
| Timeout during workflow? | Don't timeout if workflow active |
| Default mode? | HOME |
| Timeout duration? | 2 minutes |

---

*Next step: Create implementation plan and begin Phase 1*
