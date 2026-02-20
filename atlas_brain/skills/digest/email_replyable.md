---
name: digest/email_replyable
domain: digest
description: Classify whether an email expects a human reply
tags: [email, triage, classification, autonomous]
version: 1
---

# Email Replyable Classification

/no_think

You are classifying whether an email expects a human reply. Answer with exactly one word: YES or NO.

## Input

You will receive JSON with these fields:
- `sender`: sender name and email address
- `subject`: email subject line
- `body_snippet`: first 200 characters of the email body

## Rules

- NO if the sender is an automated system (service notifications, payment processors, shipping trackers, marketing platforms)
- NO if the email is a receipt, confirmation, status update, or alert that doesn't expect a reply
- NO if the sender address suggests no-reply even without an explicit noreply prefix (e.g. system@, auto@, mailer@)
- YES if the email asks a question, requests information, proposes a meeting, or initiates a conversation
- YES if the email is from a real person who clearly expects a response
- When unsure, default to YES (better to draft an unnecessary reply than miss a real email)

## Output

Respond with exactly one word: YES or NO
