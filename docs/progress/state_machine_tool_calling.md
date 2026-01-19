# State Machine Tool Calling Pattern

## Problem: LLM Tool Calling is Unreliable

Tested gpt-oss:20b with native Ollama tool calling:
- **1 tool**: ~100% success rate
- **7+ tools**: ~33% success rate
- Often asks clarifying questions instead of calling tools
- Inconsistent behavior even with same prompts

## Solution: State Machine + Rule-Based Tool Selection

The ReceptionistAgent demonstrates a reliable pattern for multi-step tool workflows.

### Core Concept

```
LLM decides WHAT to say (response generation)
State machine decides WHAT to do (tool selection)
```

The LLM is NOT used for:
- Deciding which tool to call
- Extracting parameters from user input
- Managing conversation flow

### Architecture

```
                    User Input
                        |
                        v
    +------------------------------------------+
    |              THINK PHASE                  |
    |  1. Rule-based intent detection           |
    |  2. Regex-based data extraction           |
    |  3. State machine phase transition        |
    |  4. Explicit tool selection               |
    +------------------------------------------+
                        |
                        v
    +------------------------------------------+
    |               ACT PHASE                   |
    |  1. Build params from accumulated context |
    |  2. Execute tool directly via registry    |
    |  3. Store results                         |
    +------------------------------------------+
                        |
                        v
    +------------------------------------------+
    |             RESPOND PHASE                 |
    |  LLM generates conversational response    |
    |  (only thing LLM does)                    |
    +------------------------------------------+
```

### State Machine Flow (ReceptionistAgent)

```
GREETING -----> COLLECTING -----> CONFIRMING -----> COMPLETE
    |               |                  |
    |               |                  |
    v               v                  v
 Detect          Extract            Confirm
 Intent          Data               & Book
```

**Phase Transitions:**
- GREETING -> COLLECTING: User wants to book (keyword: "estimate", "schedule", "appointment")
- COLLECTING -> CONFIRMING: Has enough info (name + address OR time)
- CONFIRMING -> COMPLETE: User confirms ("yes", "correct", "sounds good")
- CONFIRMING -> COLLECTING: User wants changes ("no", "change", "different")

### Key Components

#### 1. Context Accumulator (CallContext)
```python
@dataclass
class CallContext:
    phase: ConversationPhase = ConversationPhase.GREETING
    caller_name: Optional[str] = None
    caller_phone: Optional[str] = None  # From caller ID
    service_address: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
```

Data persists across turns. Each turn adds to it, never replaces.

#### 2. Rule-Based Intent Detection
```python
def _wants_estimate(self, text: str) -> bool:
    keywords = ["estimate", "appointment", "schedule", "book", "cleaning"]
    return any(kw in text for kw in keywords)

def _is_confirmation(self, text: str) -> bool:
    keywords = ["yes", "yeah", "correct", "sounds good", "perfect", "ok"]
    return any(kw in text for kw in keywords)
```

No LLM inference needed. Fast and deterministic.

#### 3. Regex-Based Data Extraction
```python
# Address pattern
address_pattern = r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr)'
match = re.search(address_pattern, text_lower)
if match:
    self._call_context.service_address = text[match.start():match.end()]

# Name extraction (after "name is", "I'm", etc.)
name_triggers = ["name is", "i'm", "this is", "my name's"]
```

#### 4. Explicit Tool Selection in Think Phase
```python
elif phase == ConversationPhase.CONFIRMING:
    if self._is_confirmation(query_lower):
        result.action_type = "tool_use"
        result.tools_to_call = ["book_appointment"]  # EXPLICIT
        self._call_context.phase = ConversationPhase.COMPLETE
```

Tool is selected by code logic, not LLM.

#### 5. Direct Tool Execution in Act Phase
```python
for tool_name in think_result.tools_to_call:
    if tool_name == "book_appointment":
        params = {
            "customer_name": call_ctx.caller_name,
            "customer_phone": call_ctx.caller_phone,
            "date": book_date.strftime("%Y-%m-%d"),
            "time": book_time,
            "address": call_ctx.service_address,
        }
    tool_result = await tools.execute(tool_name, params)
```

---

## Do You Need Multiple State Machines?

**Short answer: One base pattern, multiple configurations.**

### What Can Be Shared

1. **Base Agent Framework** (already exists in `base.py`)
   - Think -> Act -> Respond flow
   - Timer and logging
   - Error handling

2. **Tool Registry** (already exists)
   - All tools registered once
   - Any agent can call any tool

3. **Data Extraction Utilities** (can be shared)
   - Address regex
   - Name extraction
   - Date/time parsing (dateparser library)

### What Needs Per-Use-Case Configuration

| Use Case | Phases | Data to Collect | Tools |
|----------|--------|-----------------|-------|
| Appointment Booking | GREETING -> COLLECTING -> CONFIRMING -> COMPLETE | name, phone, address, date/time | check_availability, book_appointment |
| Reminder Setting | COLLECTING -> CONFIRMING -> COMPLETE | message, time | set_reminder |
| Calendar Query | SINGLE_TURN | date range | get_calendar |
| Weather Query | SINGLE_TURN | location (optional) | get_weather |

### Recommendation: Workflow Templates

Instead of multiple state machines, create **workflow templates**:

```python
# workflows.py

BOOKING_WORKFLOW = {
    "phases": ["greeting", "collecting", "confirming", "complete"],
    "required_data": ["customer_name", "date", "time"],
    "optional_data": ["address", "email"],
    "tools": {
        "collecting": "check_availability",
        "complete": "book_appointment",
    },
    "transitions": {
        "greeting": {"wants_booking": "collecting"},
        "collecting": {"has_enough_info": "confirming"},
        "confirming": {"confirmed": "complete", "rejected": "collecting"},
    },
}

REMINDER_WORKFLOW = {
    "phases": ["collecting", "confirming", "complete"],
    "required_data": ["message", "time"],
    "tools": {
        "complete": "set_reminder",
    },
    "transitions": {
        "collecting": {"has_enough_info": "confirming"},
        "confirming": {"confirmed": "complete"},
    },
}
```

Then a generic `WorkflowAgent` interprets these templates.

---

## When to Use State Machine vs LLM Tool Calling

### Use State Machine When:
- Multi-turn data collection (booking, forms)
- Predictable conversation flow
- High reliability required
- Latency matters (no extra LLM calls)

### Use LLM Tool Calling When:
- Single-turn queries ("what time is it")
- Unpredictable user intent
- Only 1-2 tools needed
- OK with occasional failures

### Hybrid Approach (Current AtlasAgent)
- Intent parser detects tool need -> target single tool
- `execute_with_tools()` with `target_tool` parameter
- Works for simple queries, not multi-step

---

## Test Results

**ReceptionistAgent State Machine:**
```
Turn 1: "schedule a cleaning estimate" -> COLLECTING
Turn 2: "My name is Sarah Johnson" -> COLLECTING (name extracted)
Turn 3: "789 Main Street" -> CONFIRMING (address extracted)
Turn 4: "Tomorrow morning perfect" -> COMPLETE (book_appointment executed)

Result: Google Calendar event created successfully
```

**LLM Tool Calling (for comparison):**
```
Query: "Check availability then book at 7 PM"
Result: Called check_availability, then asked clarifying questions
        instead of calling book_appointment
```

---

## Files

- `atlas_brain/agents/receptionist.py` - Full implementation
- `atlas_brain/agents/base.py` - Base agent with Think->Act->Respond
- `atlas_brain/agents/protocols.py` - AgentContext, ThinkResult, ActResult
- `atlas_brain/tools/scheduling.py` - check_availability, book_appointment tools

---

## Integration with Modes Architecture

Atlas has 5 modes defined in `atlas_brain/modes/config.py`:

| Mode | Purpose | State Machine? | Why |
|------|---------|----------------|-----|
| **HOME** | Device control | NO | Single-turn commands ("turn on lights") |
| **RECEPTIONIST** | Business scheduling | YES | Multi-turn booking, data collection |
| **COMMS** | Personal comms | MAYBE | Scheduling calls might need multi-turn |
| **SECURITY** | Cameras, alerts | NO | Mostly single queries ("show camera") |
| **CHAT** | Conversation | NO | Cloud LLM handles complexity |

### Where State Machines ARE Needed

1. **Appointment/Estimate Booking** (RECEPTIONIST mode)
   - Collect: name, address, date/time
   - Confirm before booking
   - Execute: book_appointment
   - ✅ Already implemented in ReceptionistAgent

2. **Reminder Setting with Details** (COMMS/RECEPTIONIST)
   - If user says "remind me" without details
   - Collect: message, time, recurrence
   - Confirm before setting
   - Execute: set_reminder

3. **Complex Calendar Operations**
   - "Schedule a meeting with John next week"
   - Need: check John's availability, find mutual time, confirm
   - Multiple tool calls: check_availability → book_appointment

### Where State Machines are NOT Needed

1. **Device Commands** (HOME mode)
   - "Turn on the living room lights" → Single intent → Execute
   - Fast path, no data collection needed

2. **Simple Tool Queries** (Any mode)
   - "What time is it?" → get_time → Response
   - "Weather tomorrow" → get_weather → Response
   - Single turn, target_tool filtering works fine

3. **Security Queries** (SECURITY mode)
   - "Show front door camera" → show_camera_feed
   - "Who's in the driveway?" → get_current_detections
   - Single queries, no multi-turn needed

---

## How to Integrate into AtlasAgent

AtlasAgent needs to detect when to use state machine vs fast path:

```python
# In AtlasAgent._do_think()

# 1. Check for mode switch command
mode_switch = self._mode_manager.parse_mode_switch(query)
if mode_switch:
    return ThinkResult(action_type="mode_switch", ...)

# 2. Detect workflow type
if self._is_multi_step_workflow(query):
    # Use state machine approach
    return self._think_with_state_machine(context)
else:
    # Use fast path (current implementation)
    return self._think_fast_path(context)

def _is_multi_step_workflow(self, query: str) -> bool:
    """Detect if query needs multi-turn data collection."""
    multi_step_triggers = [
        # Booking
        "book", "schedule", "appointment", "estimate",
        # Reminders with missing info
        "remind me",  # but not "remind me at 5pm to X" (has all info)
    ]
    # Check if trigger present AND missing required data
    ...
```

### Shared Context Accumulator

Instead of `CallContext` per agent, use a generic `WorkflowContext`:

```python
@dataclass
class WorkflowContext:
    """Context accumulated during multi-turn workflow."""
    workflow_type: str  # "booking", "reminder", etc.
    phase: str = "collecting"  # collecting, confirming, complete

    # Generic data storage
    collected_data: dict = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)

    # Tracking
    turns: int = 0

    def has_required_data(self) -> bool:
        return all(
            self.collected_data.get(f)
            for f in self.required_fields
        )
```

### Workflow Definitions (No Duplicate State Machines)

```python
WORKFLOWS = {
    "booking": {
        "required_fields": ["customer_name", "date", "time"],
        "optional_fields": ["address", "phone", "service_type"],
        "confirm_tool": "book_appointment",
        "extractors": {
            "customer_name": extract_name,
            "date": extract_date,
            "time": extract_time,
            "address": extract_address,
        },
    },
    "reminder": {
        "required_fields": ["message", "time"],
        "optional_fields": ["recurrence"],
        "confirm_tool": "set_reminder",
        "extractors": {
            "message": extract_reminder_message,
            "time": extract_datetime,
        },
    },
}
```

### Single Generic State Machine

```python
class WorkflowStateMachine:
    """Generic state machine for multi-turn workflows."""

    def __init__(self, workflow_type: str):
        self.config = WORKFLOWS[workflow_type]
        self.context = WorkflowContext(
            workflow_type=workflow_type,
            required_fields=self.config["required_fields"],
        )

    def process_turn(self, user_input: str) -> tuple[str, Optional[str]]:
        """
        Process user input, return (phase, tool_to_call).

        Returns:
            phase: "collecting", "confirming", or "complete"
            tool_to_call: Tool name if ready to execute, else None
        """
        # Extract data from input
        for field, extractor in self.config["extractors"].items():
            value = extractor(user_input)
            if value:
                self.context.collected_data[field] = value

        # Check phase transition
        if self.context.phase == "collecting":
            if self.context.has_required_data():
                self.context.phase = "confirming"

        elif self.context.phase == "confirming":
            if is_confirmation(user_input):
                self.context.phase = "complete"
                return "complete", self.config["confirm_tool"]
            elif is_rejection(user_input):
                self.context.phase = "collecting"

        return self.context.phase, None
```

---

## Summary: No Duplicate State Machines

1. **One generic WorkflowStateMachine class**
2. **Configuration per workflow type** (booking, reminder, etc.)
3. **Shared extractors** (date, time, name, address)
4. **Mode determines available workflows**:
   - RECEPTIONIST: booking, reminder
   - COMMS: reminder, schedule_call
   - HOME: none (fast path only)

5. **AtlasAgent detects workflow vs fast path**:
   - Multi-step trigger + missing data → State machine
   - Simple query or complete data → Fast path

---

## Test Confirmation

The ReceptionistAgent booking test proved the pattern works automatically:

```
User: "schedule a cleaning estimate"      → COLLECTING (detected booking intent)
User: "Sarah Johnson"                     → COLLECTING (extracted name)
User: "789 Main Street"                   → CONFIRMING (extracted address, has enough)
User: "Tomorrow morning would be perfect" → COMPLETE (confirmed, book_appointment executed)

Calendar event created: "Free Estimate - Sarah Johnson"
```

No manual prompting per step. Rules + data extraction + phase transitions = automatic multi-step workflow.
