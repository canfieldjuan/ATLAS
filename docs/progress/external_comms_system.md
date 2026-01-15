# External Communications System

## Status: Planning
**Started**: 2026-01-14
**Last Updated**: 2026-01-14

## Vision

Build a comprehensive AI-powered communications system that can:
- Answer and make phone calls with natural voice conversation
- Send and receive SMS/text messages
- Automatically route based on context (business vs personal)
- Schedule appointments for business customers
- Run primarily through Atlas (self-hosted where possible)

## Current Setup

- **Personal Phone**: Regular cell (details TBD)
- **Business Phone**: eVoice (Effingham Office Maids)
- **Business Type**: Cleaning service - appointment-based

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │           Telephony Provider            │
                    │    (Twilio / SignalWire / Telnyx)       │
                    │                                         │
                    │  ┌─────────────┐  ┌─────────────────┐  │
                    │  │ Voice/SIP   │  │  SMS Gateway    │  │
                    │  └──────┬──────┘  └────────┬────────┘  │
                    └─────────┼──────────────────┼───────────┘
                              │ WebSocket        │ Webhook
                              │ (audio stream)   │ (messages)
                              ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      ATLAS BRAIN                             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Telephony Service Layer                    │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │ │
│  │  │ Call Manager│  │ SMS Manager │  │Context Router │  │ │
│  │  │ - Inbound   │  │ - Inbound   │  │ - Number→Ctx  │  │ │
│  │  │ - Outbound  │  │ - Outbound  │  │ - Caller ID   │  │ │
│  │  │ - Transfer  │  │ - Threading │  │ - Business hrs│  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘  │ │
│  └─────────┼────────────────┼─────────────────┼──────────┘ │
│            │                │                 │             │
│            ▼                ▼                 ▼             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Conversation Engine                           │ │
│  │                                                         │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐ │ │
│  │  │   STT   │→ │   LLM   │→ │   TTS   │→ │  Actions  │ │ │
│  │  │(stream) │  │(context)│  │(stream) │  │(calendar) │ │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └───────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Business Contexts                             │ │
│  │                                                         │ │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐ │ │
│  │  │  Effingham Office   │  │      Personal           │ │ │
│  │  │      Maids          │  │                         │ │ │
│  │  │                     │  │  - Personal contacts    │ │ │
│  │  │  - Business hours   │  │  - Family/friends mode  │ │ │
│  │  │  - Services/pricing │  │  - Message taking       │ │ │
│  │  │  - Scheduling       │  │                         │ │ │
│  │  │  - Customer DB      │  │                         │ │ │
│  │  └─────────────────────┘  └─────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Phased Implementation

### Phase 1: Foundation (Current Focus)
- [ ] Design telephony abstraction layer (provider-agnostic)
- [ ] Define protocols for call/SMS handling
- [ ] Create context/business configuration system
- [ ] Set up webhook endpoints for receiving calls/SMS
- [ ] Basic call flow: answer → STT → LLM → TTS → respond

### Phase 2: Business Integration
- [ ] Port/forward eVoice number to programmable provider
- [ ] Effingham Office Maids context configuration
  - Business hours
  - Services offered
  - Pricing information
  - Greeting/persona
- [ ] Calendar integration for appointment scheduling
- [ ] Customer database (name, address, history)

### Phase 3: SMS Conversations
- [ ] Inbound SMS handling with context awareness
- [ ] Outbound SMS (confirmations, reminders)
- [ ] Conversation threading (multi-turn SMS)
- [ ] Appointment confirmation/reminder automation

### Phase 4: Advanced Features
- [ ] Outbound calling (appointment reminders, follow-ups)
- [ ] Call recording and transcription storage
- [ ] Voicemail with AI transcription
- [ ] Call transfer to personal phone when needed
- [ ] Multi-business support (add more contexts)

### Phase 5: Intelligence
- [ ] Customer recognition (caller ID → history)
- [ ] Sentiment analysis during calls
- [ ] Automatic quote generation
- [ ] Lead qualification
- [ ] Analytics dashboard

## Telephony Provider Comparison

| Provider   | Voice WebSocket | SMS | Pricing      | Notes                    |
|------------|-----------------|-----|--------------|--------------------------|
| Twilio     | ✓ Media Streams | ✓   | $0.013/min   | Most docs, largest       |
| SignalWire | ✓ RELAY         | ✓   | $0.010/min   | Twilio-compatible, cheaper|
| Telnyx     | ✓ WebRTC        | ✓   | $0.007/min   | Cheapest, good quality   |
| Vonage     | ✓ WebSocket     | ✓   | $0.014/min   | Good international       |

**Recommendation**: Start with Twilio (best docs) or SignalWire (cheaper, compatible API)

## Effingham Office Maids - Business Context

### Services (to be filled in)
- [ ] List of cleaning services offered
- [ ] Pricing structure
- [ ] Service areas
- [ ] Minimum booking requirements

### Scheduling Requirements
- [ ] Business hours
- [ ] Appointment duration by service type
- [ ] Buffer time between appointments
- [ ] How far out can customers book?
- [ ] Cancellation policy

### Call Handling
- Greeting: "Thank you for calling Effingham Office Maids, this is Atlas. How can I help you today?"
- Key intents:
  - Schedule appointment
  - Get quote
  - Reschedule/cancel
  - Ask questions about services
  - Speak to owner (transfer)

## Technical Considerations

### Latency Budget (Voice Calls)
For natural conversation, total round-trip should be <1 second:
- Audio chunk receive: ~100ms
- STT processing: ~200-300ms
- LLM response: ~200-400ms
- TTS synthesis: ~100-200ms
- Audio send: ~100ms

**Strategy**:
- Streaming STT (process as audio arrives)
- Fast LLM (Ollama local, or tune for speed)
- Streaming TTS (start playing before full response)

### State Management
- Call state: ringing, connected, on-hold, ended
- Conversation context: what's been discussed
- Business context: which number, what rules apply
- Action state: scheduling in progress, etc.

## Files to Create

### Phase 1 Foundation
| File | Purpose |
|------|---------|
| `atlas_brain/comms/__init__.py` | Communications module |
| `atlas_brain/comms/protocols.py` | Abstract interfaces |
| `atlas_brain/comms/config.py` | Comms configuration |
| `atlas_brain/comms/call_manager.py` | Call lifecycle management |
| `atlas_brain/comms/sms_manager.py` | SMS handling |
| `atlas_brain/comms/context.py` | Business context routing |
| `atlas_brain/comms/providers/base.py` | Provider abstraction |
| `atlas_brain/comms/providers/twilio.py` | Twilio implementation |
| `atlas_brain/api/comms.py` | Webhook endpoints |

## Questions to Resolve

1. **Number porting**: Port eVoice number to new provider, or forward calls?
2. **Personal number**: Include personal phone in system, or business only first?
3. **Calendar system**: Use existing Google Calendar, or dedicated scheduling?
4. **Customer data**: Where to store customer info? (PostgreSQL makes sense)
5. **Call recording**: Record calls for training/review? (legal considerations)

## Progress Log

### 2026-01-14
- Created initial planning document
- Defined architecture overview
- Outlined phased implementation approach
- Identified key technical considerations
