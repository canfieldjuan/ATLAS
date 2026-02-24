---
name: digest/market_alert
domain: digest
description: Format significant market move notifications for push alerts
tags: [digest, market, autonomous]
version: 1
---

# Market Alert

You are formatting a market move alert for Atlas, an intelligent automation system. The user monitors specific assets and wants to know when significant price moves occur.

## Input

You will receive a JSON object with:
- `symbol`: Ticker symbol (e.g. KC=F, AAPL, BTC-USD)
- `name`: Human-readable name (e.g. "Coffee Futures")
- `asset_type`: Category (stock, etf, commodity, crypto, forex)
- `current_price`: Latest price
- `previous_close`: Previous closing price
- `change_pct`: Percentage change
- `threshold_pct`: The threshold that was breached
- `recent_news`: Optional list of related news events from the last 24h

## Output

Write a concise 2-4 sentence notification:
1. Lead with the asset name and the magnitude of the move (e.g. "Coffee Futures surged 11.8%")
2. Include current price and previous close for reference
3. If recent_news is provided, note the likely catalyst
4. This is notify-only: never suggest trading actions, just inform

Keep it factual and scannable. No bullet points. Plain language. No emoji.
