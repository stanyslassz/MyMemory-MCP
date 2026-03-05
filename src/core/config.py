"""Configuration loader for memory-ai. Loads config.yaml + .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _resolve_path(base: Path, p: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass
class LLMStepConfig:
    model: str
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 60
    api_base: str | None = None


@dataclass
class EmbeddingsConfig:
    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    api_base: str | None = None
    chunk_size: int = 400
    chunk_overlap: int = 80


@dataclass
class ScoringConfig:
    model: str = "act_r"
    decay_factor: float = 0.5
    decay_factor_short_term: float = 0.8
    importance_weight: float = 0.3
    spreading_weight: float = 0.2
    permanent_min_score: float = 0.5
    relation_strength_base: float = 0.5
    relation_decay_halflife: int = 180
    window_size: int = 50
    min_score_for_context: float = 0.3
    # Legacy fields (used until Phase 2 replaces scoring.py)
    weight_importance: float = 0.4
    weight_frequency: float = 0.3
    weight_recency: float = 0.3
    frequency_cap: int = 20
    recency_halflife_days: int = 30


@dataclass
class FAISSConfig:
    index_path: str = "./memory/_memory.faiss"
    mapping_path: str = "./memory/_memory.pkl"
    manifest_path: str = "./memory/_faiss_manifest.json"
    top_k: int = 5


@dataclass
class CategoriesConfig:
    observations: list[str] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    folders: dict[str, str] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    doc_pipeline: bool = False


@dataclass
class IngestConfig:
    recovery_threshold_seconds: int = 300
    max_retries: int = 3
    jobs_path: str = "./_ingest_jobs.json"


@dataclass
class Config:
    user_language: str = "fr"
    llm_extraction: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b"))
    llm_arbitration: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b"))
    llm_context: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b", temperature=0.3))
    llm_consolidation: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b", temperature=0.1))
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    memory_path: Path = field(default_factory=lambda: Path("./memory"))
    context_max_tokens: int = 3000
    context_budget: dict[str, int] = field(default_factory=dict)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    prompts_path: Path = field(default_factory=lambda: Path("./prompts"))
    categories: CategoriesConfig = field(default_factory=CategoriesConfig)
    mcp_transport: str = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    job_schedule: str = "0 3 * * *"
    job_idle_trigger_minutes: int = 10
    job_max_chats_per_run: int = 20
    project_root: Path = field(default_factory=lambda: Path("."))
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)

    def get_folder_for_type(self, entity_type: str) -> str:
        """Return the memory subfolder for an entity type."""
        return self.categories.folders.get(entity_type, "interests")


def _build_llm_step(data: dict[str, Any]) -> LLMStepConfig:
    return LLMStepConfig(
        model=data.get("model", "ollama/llama3.1:8b"),
        temperature=data.get("temperature", 0.0),
        max_retries=data.get("max_retries", 3),
        timeout=data.get("timeout", 60),
        api_base=data.get("api_base"),
    )


def load_config(config_path: str | Path | None = None, project_root: Path | None = None) -> Config:
    """Load config from yaml file + .env. Returns a Config dataclass."""
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root).resolve()

    # Load .env first (so API keys are available to LiteLLM)
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if config_path is None:
        config_path = project_root / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        return Config(project_root=project_root)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    llm = raw.get("llm", {})
    emb = raw.get("embeddings", {})
    mem = raw.get("memory", {})
    scoring = raw.get("scoring", {})
    faiss_cfg = raw.get("faiss", {})
    cats = raw.get("categories", {})
    job = raw.get("job", {})
    feat = raw.get("features", {})
    ingest_cfg = raw.get("ingest", {})

    memory_path = _resolve_path(project_root, mem.get("path", "./memory"))
    prompts_path = _resolve_path(project_root, raw.get("prompts", {}).get("path", "./prompts"))

    return Config(
        user_language=raw.get("user_language", "fr"),
        llm_extraction=_build_llm_step(llm.get("extraction", {})),
        llm_arbitration=_build_llm_step(llm.get("arbitration", {})),
        llm_context=_build_llm_step(llm.get("context", {})),
        llm_consolidation=_build_llm_step(llm.get("consolidation", {})),
        embeddings=EmbeddingsConfig(
            provider=emb.get("provider", "sentence-transformers"),
            model=emb.get("model", "all-MiniLM-L6-v2"),
            api_base=emb.get("api_base"),
            chunk_size=emb.get("chunk_size", 400),
            chunk_overlap=emb.get("chunk_overlap", 80),
        ),
        memory_path=memory_path,
        context_max_tokens=mem.get("context_max_tokens", 3000),
        context_budget=mem.get("context_budget", {}),
        scoring=ScoringConfig(
            model=scoring.get("model", "act_r"),
            decay_factor=scoring.get("decay_factor", 0.5),
            decay_factor_short_term=scoring.get("decay_factor_short_term", 0.8),
            importance_weight=scoring.get("importance_weight", 0.3),
            spreading_weight=scoring.get("spreading_weight", 0.2),
            permanent_min_score=scoring.get("permanent_min_score", 0.5),
            relation_strength_base=scoring.get("relation_strength_base", 0.5),
            relation_decay_halflife=scoring.get("relation_decay_halflife", 180),
            window_size=scoring.get("window_size", 50),
            min_score_for_context=scoring.get("min_score_for_context", 0.3),
            weight_importance=scoring.get("weight_importance", 0.4),
            weight_frequency=scoring.get("weight_frequency", 0.3),
            weight_recency=scoring.get("weight_recency", 0.3),
            frequency_cap=scoring.get("frequency_cap", 20),
            recency_halflife_days=scoring.get("recency_halflife_days", 30),
        ),
        faiss=FAISSConfig(
            index_path=str(_resolve_path(project_root, faiss_cfg.get("index_path", "./memory/_memory.faiss"))),
            mapping_path=str(_resolve_path(project_root, faiss_cfg.get("mapping_path", "./memory/_memory.pkl"))),
            manifest_path=str(_resolve_path(project_root, faiss_cfg.get("manifest_path", "./memory/_faiss_manifest.json"))),
            top_k=faiss_cfg.get("top_k", 5),
        ),
        prompts_path=prompts_path,
        categories=CategoriesConfig(
            observations=cats.get("observations", []),
            entity_types=cats.get("entity_types", []),
            relation_types=cats.get("relation_types", []),
            folders=cats.get("folders", {}),
        ),
        mcp_transport=raw.get("mcp", {}).get("transport", "stdio"),
        mcp_host=raw.get("mcp", {}).get("host", "127.0.0.1"),
        mcp_port=raw.get("mcp", {}).get("port", 8000),
        job_schedule=job.get("schedule", "0 3 * * *"),
        job_idle_trigger_minutes=job.get("idle_trigger_minutes", 10),
        job_max_chats_per_run=job.get("max_chats_per_run", 20),
        project_root=project_root,
        features=FeaturesConfig(
            doc_pipeline=feat.get("doc_pipeline", False),
        ),
        ingest=IngestConfig(
            recovery_threshold_seconds=ingest_cfg.get("recovery_threshold_seconds", 300),
            max_retries=ingest_cfg.get("max_retries", 3),
            jobs_path=str(_resolve_path(project_root, ingest_cfg.get("jobs_path", "./memory/_ingest_jobs.json"))),
        ),
    )
