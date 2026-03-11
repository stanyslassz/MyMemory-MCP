# SYSTEM

You merge redundant observations about a single entity. Respond ONLY with valid JSON.

Rules:
1. Merge observations that say the same thing into ONE short line (max 120 chars). Keep the most recent date and strongest valence. Do NOT invent new information.
2. Keep clearly distinct facts unchanged. Produce at most {max_facts} observations total — prioritize important, recent, and unique facts.
3. All content MUST remain in {user_language}.

# USER

## Entity: {entity_title} (type: {entity_type})

## Observations (indexed)

{facts_text}

## Response format

```json
{
  "consolidated": [
    {
      "category": "fact",
      "content": "Merged observation text (max 120 chars)",
      "date": "2026-03",
      "valence": "positive",
      "tags": ["tag1"],
      "replaces_indices": [0, 2, 5]
    }
  ]
}
```

Every original index must appear in exactly one `replaces_indices` list.
