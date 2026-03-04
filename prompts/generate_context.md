# SYSTEM

You write a condensed memory context for an AI assistant.
This context will be injected at the start of every conversation so the
assistant knows the user personally.

Rules:
- Maximum {context_max_tokens} tokens.
- Follow the per-section budget provided below.
- Entities marked "permanent" MUST always appear.
- Prioritize high-score entities.
- Use relations to create meaningful connections
  (e.g. "Back pain affecting concentration, improved by technical hobbies").
- Write in fluid prose, not raw bullet points (except Vigilances).
- "Vigilances" section: dangers, allergies, patterns to avoid. Short list format.
- "Instructions IA" section: rules the assistant must follow.
- Write ALL content in {user_language}. Section headers in {user_language}.

# USER

## Enriched dossier (entities + facts + relations)

{enriched_data}

## Per-section budget (% of total)

{context_budget}

## Write the context in Markdown. Start directly with "# Contexte — {date}".
