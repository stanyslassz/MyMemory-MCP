"""LLM abstraction layer. Loads prompts from .md files, calls LiteLLM + Instructor."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar

import instructor
import litellm
from pydantic import BaseModel

from src.core.config import Config, LLMStepConfig
from src.core.models import EntityResolution, RawExtraction

T = TypeVar("T", bound=BaseModel)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> tags emitted by some models (Qwen3, DeepSeek-R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_prompt(name: str, config: Config, **variables: Any) -> str:
    """Load a prompt from prompts/{name}.md and replace {variables}.

    Always injects {user_language} from config.
    Uses str.replace() per variable to avoid conflicts with JSON braces.
    """
    prompt_path = config.prompts_path / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8")

    # Always inject user_language
    variables.setdefault("user_language", config.user_language)

    # Inject category lists if not provided
    variables.setdefault("categories_observations", ", ".join(config.categories.observations))
    variables.setdefault("categories_entity_types", ", ".join(config.categories.entity_types))
    variables.setdefault("categories_relation_types", ", ".join(config.categories.relation_types))

    for key, value in variables.items():
        content = content.replace(f"{{{key}}}", str(value))

    return content


def _get_client(step_config: LLMStepConfig) -> instructor.Instructor:
    """Create an Instructor-patched client for a specific LLM step."""
    kwargs: dict[str, Any] = {}
    if step_config.api_base:
        kwargs["api_base"] = step_config.api_base

    client = instructor.from_litellm(
        litellm.completion,
        mode=instructor.Mode.JSON,
    )
    return client


def _call_structured(
    step_config: LLMStepConfig,
    prompt: str,
    response_model: type[T],
) -> T:
    """Make a structured LLM call with Instructor validation."""
    client = _get_client(step_config)

    kwargs: dict[str, Any] = {
        "model": step_config.model,
        "messages": [{"role": "user", "content": prompt}],
        "response_model": response_model,
        "max_retries": step_config.max_retries,
        "temperature": step_config.temperature,
    }
    if step_config.timeout:
        kwargs["timeout"] = step_config.timeout
    if step_config.api_base:
        kwargs["api_base"] = step_config.api_base

    return client.chat.completions.create(**kwargs)


def call_extraction(chat_content: str, config: Config) -> RawExtraction:
    """Step 1: Extract facts and entities from a chat conversation."""
    schema = json.dumps(RawExtraction.model_json_schema(), indent=2)
    prompt = load_prompt(
        "extract_facts",
        config,
        chat_content=chat_content,
        json_schema=schema,
    )
    return _call_structured(config.llm_extraction, prompt, RawExtraction)


def call_arbitration(
    entity_name: str,
    entity_context: str,
    candidates: list[dict],
    config: Config,
) -> EntityResolution:
    """Step 3: Arbitrate ambiguous entity resolution."""
    schema = json.dumps(EntityResolution.model_json_schema(), indent=2)
    candidates_str = "\n".join(
        f"- {c['id']}: {c['title']} (type: {c['type']}, aliases: {c.get('aliases', [])})"
        for c in candidates
    )
    prompt = load_prompt(
        "arbitrate_entity",
        config,
        entity_name=entity_name,
        entity_context=entity_context,
        candidates=candidates_str,
        json_schema=schema,
    )
    return _call_structured(config.llm_arbitration, prompt, EntityResolution)


def call_context_generation(enriched_data: str, config: Config) -> str:
    """Step 7: Generate _context.md from enriched data. Returns free-text markdown."""
    from datetime import date

    budget_str = "\n".join(f"- {k}: {v}%" for k, v in config.context_budget.items())
    prompt = load_prompt(
        "generate_context",
        config,
        context_max_tokens=str(config.context_max_tokens),
        enriched_data=enriched_data,
        context_budget=budget_str,
        date=date.today().isoformat(),
    )

    step_config = config.llm_context
    kwargs: dict[str, Any] = {
        "model": step_config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": step_config.temperature,
    }
    if step_config.timeout:
        kwargs["timeout"] = step_config.timeout
    if step_config.api_base:
        kwargs["api_base"] = step_config.api_base

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content or ""
    return strip_thinking(text)
