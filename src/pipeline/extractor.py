"""Step 1: Extract facts and entities from chat conversations using LLM."""

from __future__ import annotations

from src.core.config import Config
from src.core.llm import call_extraction
from src.core.models import RawExtraction


def extract_from_chat(chat_content: str, config: Config) -> RawExtraction:
    """Extract structured information from a chat conversation.

    Uses the LLM via Instructor to return a validated RawExtraction.
    """
    if not chat_content.strip():
        return RawExtraction(entities=[], relations=[], summary="")

    return call_extraction(chat_content, config)
