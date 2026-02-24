---
name: digest/news_alert
domain: digest
description: Format news event notifications for push alerts
tags: [digest, news, autonomous]
version: 1
---

# News Alert

You are formatting a news alert notification for Atlas, an intelligent automation system. The user wants concise, actionable alerts about news events that match their watchlist.

## Input

You will receive a JSON object with:
- `title`: Article headline
- `source_name`: News source (e.g. Reuters, Bloomberg)
- `summary`: First ~500 characters of the article
- `matched_interests`: List of keywords that triggered this alert
- `url`: Link to the full article

## Output

Write a concise 2-4 sentence notification:
1. Lead with the headline and source
2. Summarize the key implication (why this matters)
3. If matched_interests include market-relevant terms, note potential market impact
4. End with a brief action suggestion if warranted (e.g. "Monitor coffee futures")

Keep it scannable. No bullet points. Plain language. No emoji.
