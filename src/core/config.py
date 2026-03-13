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
    context_window: int = 8192


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
    relation_strength_growth: float = 0.05
    relation_ltd_halflife: int = 360
    relation_decay_power: float = 0.3
    retrieval_threshold: float = 0.05
    emotional_boost_weight: float = 0.15
    window_size: int = 50
    min_score_for_context: float = 0.3
    max_spreading_neighbors: int = 10
    ltd_onset_days: int = 90
    min_relation_strength: float = 0.1


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
class NLPConfig:
    enabled: bool = True
    model: str = "fr_core_news_sm"
    dedup_threshold: float = 0.85
    date_extraction: bool = True
    pre_ner: bool = True


@dataclass
class SearchConfig:
    hybrid_enabled: bool = True
    rrf_k: int = 60
    weight_semantic: float = 0.5
    weight_keyword: float = 0.3
    weight_actr: float = 0.2
    fts_db_path: str = "_memory_fts.db"
    resolver_threshold: float = 0.75
    linear_faiss_weight: float = 0.6
    linear_actr_weight: float = 0.4


@dataclass
class DreamConfig:
    faiss_merge_threshold: float = 0.80
    faiss_merge_max_candidates: int = 20
    dossier_max_facts: int = 3
    prune_score_threshold: float = 0.1
    prune_min_age_days: int = 90
    prune_max_frequency: int = 1
    transitive_min_strength: float = 0.4
    transitive_max_new: int = 20


@dataclass
class FactTTLConfig:
    """TTL (time-to-live) in days per observation category. 0 = never expires."""
    context: int = 60
    project: int = 90
    emotion: int = 30
    progression: int = 60
    fact: int = 0          # permanent
    preference: int = 0
    diagnosis: int = 0
    vigilance: int = 0
    treatment: int = 0
    technique: int = 0
    decision: int = 90
    skill: int = 0
    rule: int = 0
    ai_style: int = 0
    user_reaction: int = 60
    interaction_rule: int = 0
    interpersonal: int = 0


@dataclass
class ContextConfig:
    """Parameters for context generation (builder + formatter)."""
    reserved_tokens_natural: int = 500
    reserved_tokens_structured: int = 500
    min_budget_tokens: int = 500
    top_entities_count: int = 50
    available_entities_limit: int = 30
    default_budget_pct: int = 20
    top_of_mind_limit: int = 15
    rag_chunk_preview_len: int = 200
    max_rag_results: int = 15
    max_vigilance_items: int = 15
    max_facts_per_category: int = 5
    max_facts_per_category_ai_self: int = 3
    fact_dedup_threshold: float = 0.35
    min_rel_strength: float = 0.3
    max_rel_age_days: int = 365
    history_recent_days: int = 30


@dataclass
class Config:
    user_language: str = "fr"
    llm_extraction: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b"))
    llm_arbitration: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b"))
    llm_context: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b", temperature=0.3))
    llm_consolidation: LLMStepConfig = field(default_factory=lambda: LLMStepConfig(model="ollama/llama3.1:8b", temperature=0.1))
    llm_dream: LLMStepConfig | None = None  # Optional — falls back to llm_context
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
    nlp: NLPConfig = field(default_factory=NLPConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    dream: DreamConfig = field(default_factory=DreamConfig)
    fact_ttl: FactTTLConfig = field(default_factory=FactTTLConfig)
    ctx: ContextConfig = field(default_factory=ContextConfig)
    context_llm_sections: bool = False
    context_format: str = "structured"  # "structured" (default) or "natural" (Claude Chat-like)
    max_facts: dict[str, int] = field(default_factory=lambda: {"default": 50, "ai_self": 20})

    @property
    def user_language_name(self) -> str:
        """Map language code to full name for prompts."""
        names = {"fr": "French", "en": "English", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese"}
        return names.get(self.user_language, self.user_language)

    @property
    def llm_dream_effective(self) -> LLMStepConfig:
        """Dream LLM config: uses llm_dream if set, otherwise falls back to llm_context."""
        return self.llm_dream if self.llm_dream is not None else self.llm_context

    def get_folder_for_type(self, entity_type: str) -> str:
        """Return the memory subfolder for an entity type."""
        return self.categories.folders.get(entity_type, "interests")

    def get_max_facts(self, entity_type: str) -> int:
        """Return the max facts limit for an entity type."""
        return self.max_facts.get(entity_type, self.max_facts.get("default", 50))

    def get_fact_ttl(self, category: str) -> int:
        """Return TTL in days for a fact category. 0 = never expires."""
        return getattr(self.fact_ttl, category, 0)


def _build_llm_step(data: dict[str, Any]) -> LLMStepConfig:
    return LLMStepConfig(
        model=data.get("model", "ollama/llama3.1:8b"),
        temperature=data.get("temperature", 0.0),
        max_retries=data.get("max_retries", 3),
        timeout=data.get("timeout", 60),
        api_base=data.get("api_base"),
        context_window=data.get("context_window", 8192),
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
    nlp_cfg = raw.get("nlp", {})
    search_cfg = raw.get("search", {})
    ttl_cfg = raw.get("fact_ttl", {})
    ctx_cfg = raw.get("context", {})


    memory_path = _resolve_path(project_root, mem.get("path", "./memory"))
    prompts_path = _resolve_path(project_root, raw.get("prompts", {}).get("path", "./prompts"))

    return Config(
        user_language=raw.get("user_language", "fr"),
        llm_extraction=_build_llm_step(llm.get("extraction", {})),
        llm_arbitration=_build_llm_step(llm.get("arbitration", {})),
        llm_context=_build_llm_step(llm.get("context", {})),
        llm_consolidation=_build_llm_step(llm.get("consolidation", {})),
        llm_dream=_build_llm_step(llm["dream"]) if "dream" in llm else None,
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
        context_llm_sections=mem.get("context_llm_sections", False),
        context_format=mem.get("context_format", "structured"),
        max_facts=raw.get("max_facts", {"default": 50, "ai_self": 20}),
        scoring=ScoringConfig(
            model=scoring.get("model", "act_r"),
            decay_factor=scoring.get("decay_factor", 0.5),
            decay_factor_short_term=scoring.get("decay_factor_short_term", 0.8),
            importance_weight=scoring.get("importance_weight", 0.3),
            spreading_weight=scoring.get("spreading_weight", 0.2),
            permanent_min_score=scoring.get("permanent_min_score", 0.5),
            relation_strength_base=scoring.get("relation_strength_base", 0.5),
            relation_decay_halflife=scoring.get("relation_decay_halflife", 180),
            relation_strength_growth=scoring.get("relation_strength_growth", 0.05),
            relation_ltd_halflife=scoring.get("relation_ltd_halflife", 360),
            relation_decay_power=scoring.get("relation_decay_power", 0.3),
            retrieval_threshold=scoring.get("retrieval_threshold", 0.05),
            emotional_boost_weight=scoring.get("emotional_boost_weight", 0.15),
            window_size=scoring.get("window_size", 50),
            min_score_for_context=scoring.get("min_score_for_context", 0.3),
            max_spreading_neighbors=scoring.get("max_spreading_neighbors", 10),
            ltd_onset_days=scoring.get("ltd_onset_days", 90),
            min_relation_strength=scoring.get("min_relation_strength", 0.1),
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
        nlp=NLPConfig(
            enabled=nlp_cfg.get("enabled", True),
            model=nlp_cfg.get("model", "fr_core_news_sm"),
            dedup_threshold=nlp_cfg.get("dedup_threshold", 0.85),
            date_extraction=nlp_cfg.get("date_extraction", True),
            pre_ner=nlp_cfg.get("pre_ner", True),
        ),
        search=SearchConfig(
            hybrid_enabled=search_cfg.get("hybrid_enabled", True),
            rrf_k=search_cfg.get("rrf_k", 60),
            weight_semantic=search_cfg.get("weight_semantic", 0.5),
            weight_keyword=search_cfg.get("weight_keyword", 0.3),
            weight_actr=search_cfg.get("weight_actr", 0.2),
            fts_db_path=search_cfg.get("fts_db_path", "_memory_fts.db"),
            resolver_threshold=search_cfg.get("resolver_threshold", 0.75),
            linear_faiss_weight=search_cfg.get("linear_faiss_weight", 0.6),
            linear_actr_weight=search_cfg.get("linear_actr_weight", 0.4),
        ),
        dream=DreamConfig(**{k: v for k, v in raw.get("dream", {}).items() if hasattr(DreamConfig, k)}),
        fact_ttl=FactTTLConfig(**{k: v for k, v in ttl_cfg.items() if hasattr(FactTTLConfig, k)}),
        ctx=ContextConfig(**{k: v for k, v in ctx_cfg.items() if hasattr(ContextConfig, k)}),
    )
