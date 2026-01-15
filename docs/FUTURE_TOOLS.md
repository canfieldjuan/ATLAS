# Atlas Future Tools Roadmap

## Implemented
- [x] get_time - Current time/date/timezone
- [x] get_weather - Open-Meteo weather data
- [x] get_traffic - TomTom traffic (requires API key)
- [x] get_location - Home Assistant device tracker
- [x] get_calendar - Google Calendar events
- [x] set_reminder - Natural language reminders with PostgreSQL + asyncio scheduler
- [x] list_reminders - Show upcoming reminders
- [x] complete_reminder - Mark reminder as done

## Known Issues / Improvements

### Calendar Tool
- [ ] **Multiple calendars with similar names** - When user has multiple calendars with same prefix (e.g., "Effingham - Residential", "Effingham - Commercial"), generic query like "effingham calendar" matches first one found
  - Potential fix: Fuzzy matching with scoring, prefer more specific matches
  - Potential fix: Ask user to clarify if multiple matches found
- [ ] **Calendar name discovery** - User may not know exact calendar names
  - Potential fix: Add "list my calendars" command
  - Potential fix: Return calendar name in response so user learns correct names
- [ ] **Primary calendar often empty** - Default "primary" calendar may have no events while other calendars are full
  - Current behavior: Queries all calendars by default (slower but complete)
  - Potential fix: Configure preferred calendars in env

## Planned

### High Priority
- [ ] **Music Control** - Spotify/YouTube Music
  - Play/pause/skip/volume
  - Search and play by name
  - Queue management

- [ ] **Notes** - Quick voice notes
  - "Note: pick up groceries"
  - List/search notes
  - Storage: PostgreSQL + optional sync

### Medium Priority
- [ ] **Web Search** - Brave Search API or SearXNG
  - Answer questions requiring current info
  - Return concise summaries

- [ ] **Email** - Gmail/IMAP
  - Read unread count
  - Read recent emails
  - Send quick replies

- [ ] **Contacts** - Google Contacts / CardDAV
  - Lookup phone numbers
  - "Call John" resolution

- [ ] **Timer** - Countdown timers
  - "Set a timer for 10 minutes"
  - Multiple named timers
  - Audio notification on completion

### Lower Priority
- [ ] **Shopping List** - Shared family list
  - Add/remove items
  - Sync with Todoist/AnyList

- [ ] **News** - Headlines briefing
  - Configurable sources
  - Category filtering

- [ ] **Sports** - Scores and schedules
  - "Did the Lakers win?"
  - Upcoming games

- [ ] **Stocks** - Portfolio tracking
  - Current prices
  - Daily change summary

## Design Principles for Tools

1. **Lowest Latency First**
   - Cache aggressively
   - Pre-fetch predictable data
   - Async all I/O

2. **Graceful Degradation**
   - Return cached data if API fails
   - Meaningful error messages

3. **Minimal Dependencies**
   - Prefer stdlib + httpx
   - No heavy SDKs unless necessary

4. **Privacy First**
   - Local processing when possible
   - No external telemetry
