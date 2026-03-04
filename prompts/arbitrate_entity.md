# SYSTEM

You resolve entity ambiguities in a personal memory system.
Respond ONLY with valid JSON. No text before or after.

# USER

New mention detected: "{entity_name}"
Context of appearance: "{entity_context}"

Existing entities that might match:
{candidates}

Does this mention correspond to an existing entity?
If yes, indicate which one. If no, indicate the type of the new entity.

Allowed entity types: {categories_entity_types}

## Response format (JSON)

{json_schema}
