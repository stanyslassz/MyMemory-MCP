"""LLM abstraction layer. Loads prompts from .md files, calls LiteLLM + Instructor."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, TypeVar

import instructor
import litellm
from pydantic import BaseModel

from src.core.config import Config, LLMStepConfig
from src.core.models import EntityResolution, FactConsolidation, RawExtraction

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


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

    # Use MD_JSON mode: extracts JSON from markdown code blocks.
    # Avoids response_format param that LM Studio models (e.g. gpt-oss-20b) don't support.
    client = instructor.from_litellm(
        litellm.completion,
        mode=instructor.Mode.MD_JSON,
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


class StallError(TimeoutError):
    """Raised when LLM streaming stalls (no progress for stall_timeout seconds)."""
    pass


def _call_with_stall_detection(
    step_config: LLMStepConfig,
    prompt: str,
    response_model: type[T],
    stall_timeout: int = 30,
) -> T:
    """Call LLM with stall detection: timeout only fires on real stalls.

    A stall is defined as no new tokens received for `stall_timeout` seconds.
    Active streaming resets the watchdog, so long but progressing responses
    are never killed.
    """
    result: T | None = None
    error: Exception | None = None
    last_activity = time.monotonic()
    lock = threading.Lock()
    done = threading.Event()

    def _do_call():
        nonlocal result, error, last_activity
        try:
            client = _get_client(step_config)
            kwargs: dict[str, Any] = {
                "model": step_config.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_model": response_model,
                "max_retries": step_config.max_retries,
                "temperature": step_config.temperature,
                "stream": True,
            }
            if step_config.api_base:
                kwargs["api_base"] = step_config.api_base
            # Use a generous connection timeout but no overall read timeout —
            # the watchdog thread handles stall detection instead.
            kwargs["timeout"] = step_config.timeout * 3

            partial = client.chat.completions.create_partial(**kwargs)
            chunk_count = 0
            for chunk in partial:
                with lock:
                    delta = time.monotonic() - last_activity
                    last_activity = time.monotonic()
                    chunk_count += 1
                chunk_str = str(chunk)
                has_think = "<think>" in chunk_str or "</think>" in chunk_str
                logger.debug(
                    "chunk #%d: %d chars, think=%s, delta=%.1fs",
                    chunk_count, len(chunk_str), has_think, delta,
                )
                result = chunk  # keep last complete partial
            # Final result is the last chunk (fully validated by Instructor)
        except Exception as e:
            error = e
        finally:
            done.set()

    worker = threading.Thread(target=_do_call, daemon=True)
    worker.start()

    # Watchdog: check for stalls
    while not done.is_set():
        done.wait(timeout=2.0)
        if done.is_set():
            break
        with lock:
            idle = time.monotonic() - last_activity
        if idle > stall_timeout:
            logger.warning(
                "LLM stall detected: no tokens for %.0fs (threshold=%ds)",
                idle, stall_timeout,
            )
            error = StallError(
                f"LLM stalled: no progress for {idle:.0f}s "
                f"(stall_timeout={stall_timeout}s)"
            )
            done.set()
            break

    if error is not None:
        raise error
    if result is None:
        raise StallError("LLM call produced no output")
    return result


def call_extraction(chat_content: str, config: Config) -> RawExtraction:
    """Step 1: Extract facts and entities from a chat conversation.

    Uses stall-aware streaming: active token production resets the watchdog,
    so only real stalls trigger a timeout — not slow but progressing responses.
    """
    schema = json.dumps(RawExtraction.model_json_schema(), indent=2)
    prompt = load_prompt(
        "extract_facts",
        config,
        chat_content=chat_content,
        json_schema=schema,
    )
    stall_timeout = config.llm_extraction.timeout  # reuse timeout as stall threshold
    return _call_with_stall_detection(
        config.llm_extraction, prompt, RawExtraction, stall_timeout=stall_timeout,
    )


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


def call_fact_consolidation(
    entity_title: str,
    entity_type: str,
    facts_text: str,
    config: Config,
) -> FactConsolidation:
    """Consolidate redundant observations for an entity via LLM."""
    prompt = load_prompt(
        "consolidate_facts",
        config,
        entity_title=entity_title,
        entity_type=entity_type,
        facts_text=facts_text,
    )
    return _call_structured(config.llm_context, prompt, FactConsolidation)


def call_entity_summary(
    title: str,
    entity_type: str,
    facts: list[str],
    relations: list[str],
    tags: list[str],
    config: Config,
) -> str:
    """Generate a 1-3 sentence summary for an entity. Returns free text."""
    facts_str = "\n".join(f"- {f}" for f in facts) if facts else "None"
    relations_str = "\n".join(f"- {r}" for r in relations) if relations else "None"
    tags_str = ", ".join(tags) if tags else "None"

    prompt = load_prompt(
        "summarize_entity",
        config,
        entity_title=title,
        entity_type=entity_type,
        entity_facts=facts_str,
        entity_relations=relations_str,
        entity_tags=tags_str,
    )

    step_config = config.llm_context  # Reuse context LLM config
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
    return strip_thinking(text).strip()
