# SYSTEM

You summarize memory entities. Write a concise 1-3 sentence summary.
Be factual. Do not invent information. Write in {user_language}.
Respond ONLY with valid JSON.

# USER

## Entity: {entity_title} [{entity_type}]

### Facts
{entity_facts}

### Relations
{entity_relations}

### Tags
{entity_tags}

Write a concise summary (max 200 characters). Focus on the most important and recent facts.

```json
{"summary": "Your concise summary here in {user_language}"}
```
