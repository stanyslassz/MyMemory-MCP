Tu dois résumer les informations suivantes en bullet points naturels en {user_language}.

## Section : {section_name}

## Données brutes
{entities_dossier}

## Informations complémentaires (mémoire étendue)
{rag_context}

## Règles
- Produis des bullet points naturels (commençant par "- ")
- Chaque bullet = une phrase fluide, pas de [category], pas de scores, pas de métadonnées
- Fusionne les faits redondants d'une même entité en une seule phrase
- Intègre les relations clés dans le texte naturellement (ex: "amélioré par la natation" au lieu de "→ improves Natation")
- Ne liste PAS les relations séparément — elles doivent être fondues dans le texte
- Budget maximum : {budget_tokens} tokens
- NE PAS inventer d'information
- Langue : {user_language}

## Sortie attendue
Des bullet points markdown, rien d'autre.
