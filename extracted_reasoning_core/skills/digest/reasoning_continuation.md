You are extending an in-progress reasoning synthesis with a new event.

You receive:
- prior_summary: the conclusion reached in the prior synthesis pass
- prior_claims: the supporting claims, each with confidence and source ids
- prior_confidence: the overall confidence of the prior synthesis
- event: a new event with event_type and a list of new evidence items
- witness_context: any compressed witness context the host has supplied

For each prior claim, decide whether the new event:
- reinforces it (evidence aligns; keep the claim, optionally raise confidence)
- refines it (claim is partially correct; update wording or scope)
- contradicts it (evidence undermines the claim; replace with a corrected one)

If the new event introduces a wholly new claim that was not represented
before, add it. Drop a claim only when the new evidence makes it clearly
wrong and there is no refined version to keep.

Return ONLY a JSON object with these keys:
- summary (string): the post-event synthesis summary, replacing prior_summary
- claims (array of objects): the post-event claims, each with:
    - claim (string)
    - confidence (string or number 0-1)
    - source_ids (array of strings)
    - revised_from (string or null): the prior claim text or id this
      replaces, or null for newly added claims
- confidence (number 0-1): overall confidence in the post-event synthesis

Do not return a delta. Return the complete refined view.
Do not include any prose outside the JSON object.
