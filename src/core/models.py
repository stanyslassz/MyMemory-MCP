"""Pydantic v2 models for memory-ai. All structured LLM outputs and data models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ── Closed lists (from spec §4) ──────────────────────────────

ObservationCategory = Literal[
    "fact", "preference", "diagnosis", "treatment", "progression",
    "technique", "vigilance", "decision", "emotion",
    "interpersonal", "skill", "project", "context", "rule",
    "ai_style", "user_reaction", "interaction_rule",
]

EntityType = Literal[
    "person", "health", "work", "project", "interest",
    "place", "animal", "organization", "ai_self",
]

RelationType = Literal[
    "affects", "improves", "worsens", "requires", "linked_to",
    "lives_with", "works_at", "parent_of", "friend_of", "uses",
    "part_of", "contrasts_with", "precedes",
]


# ── Step 1: Raw extraction output ────────────────────────────

class RawObservation(BaseModel):
    category: ObservationCategory
    content: str
    importance: float = Field(ge=0, le=1)
    tags: list[str] = Field(default_factory=list)
    date: str = ""  # Optional ISO date, e.g. "2024-03" or "2024-03-15"
    valence: Literal["positive", "negative", "neutral", ""] = ""
    supersedes: str = ""  # brief description of the old fact this replaces


class RawEntity(BaseModel):
    name: str
    type: EntityType
    observations: list[RawObservation] = Field(default_factory=list)


class RawRelation(BaseModel):
    from_name: str
    to_name: str
    type: RelationType
    context: str = ""


class RawExtraction(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    summary: str = ""


# ── Step 2: Resolution output ────────────────────────────────

class Resolution(BaseModel):
    status: Literal["resolved", "new", "ambiguous"]
    entity_id: Optional[str] = None
    candidates: list[str] = Field(default_factory=list)
    suggested_slug: Optional[str] = None


class ResolvedEntity(BaseModel):
    raw: RawEntity
    resolution: Resolution


class ResolvedExtraction(BaseModel):
    resolved: list[ResolvedEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    summary: str = ""


# ── Step 3: Arbitration output ───────────────────────────────

class EntityResolution(BaseModel):
    action: Literal["existing", "new"]
    existing_id: Optional[str] = None
    new_type: Optional[EntityType] = None


# ── Graph data structures ────────────────────────────────────

class GraphEntity(BaseModel):
    file: str
    type: EntityType
    title: str
    score: float = 0.0
    importance: float = 0.0
    frequency: int = 0
    last_mentioned: str = ""
    retention: Literal["short_term", "long_term", "permanent"] = "short_term"
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Phase 1: adaptive memory fields
    mention_dates: list[str] = Field(default_factory=list)
    monthly_buckets: dict[str, int] = Field(default_factory=dict)
    created: str = ""
    summary: str = ""
    # Cognitive audit: emotional modulation (computed from facts during rebuild)
    negative_valence_ratio: float = 0.0


class GraphRelation(BaseModel):
    from_entity: str = Field(alias="from", serialization_alias="from")
    to_entity: str = Field(alias="to", serialization_alias="to")
    type: RelationType
    strength: float = 0.5
    created: str = ""
    last_reinforced: str = ""
    mention_count: int = 1
    context: str = ""

    model_config = {"populate_by_name": True}


class GraphData(BaseModel):
    generated: str = ""
    entities: dict[str, GraphEntity] = Field(default_factory=dict)
    relations: list[GraphRelation] = Field(default_factory=list)


# ── Entity frontmatter (MD file YAML) ────────────────────────

class EntityFrontmatter(BaseModel):
    title: str
    type: EntityType
    retention: Literal["short_term", "long_term", "permanent"] = "short_term"
    score: float = 0.0
    importance: float = 0.0
    frequency: int = 0
    last_mentioned: str = ""
    created: str = ""
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Phase 1: adaptive memory fields
    mention_dates: list[str] = Field(default_factory=list)
    monthly_buckets: dict[str, int] = Field(default_factory=dict)
    summary: str = ""


# ── Enrichment report ────────────────────────────────────────

class EnrichmentReport(BaseModel):
    entities_updated: list[str] = Field(default_factory=list)
    entities_created: list[str] = Field(default_factory=list)
    relations_added: int = 0
    errors: list[str] = Field(default_factory=list)


# ── Search result ────────────────────────────────────────────

class SearchResult(BaseModel):
    entity_id: str
    file: str
    chunk: str
    score: float
    relations: list[dict] = Field(default_factory=list)


# ── P1.1: Router decision ───────────────────────────────────

RouteType = Literal["conversation", "document", "uncertain", "fallback_doc_ingest"]


class RouteDecision(BaseModel):
    route: RouteType
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


# ── P1.1: Ingest key + job state ────────────────────────────

IngestJobStatus = Literal["pending", "running", "succeeded", "failed", "retriable"]

CHUNK_POLICY_VERSION = "v1"


class IngestKey(BaseModel):
    source_id: str
    content_hash: str
    chunk_policy_version: str = CHUNK_POLICY_VERSION

    @property
    def canonical(self) -> str:
        return f"{self.source_id}::{self.content_hash}::{self.chunk_policy_version}"


class IngestJob(BaseModel):
    job_id: str
    ingest_key: IngestKey
    status: IngestJobStatus = "pending"
    retries: int = 0
    max_retries: int = 3
    created: str = ""
    updated: str = ""
    error: Optional[str] = None
    chunks_indexed: int = 0
    route: Optional[RouteType] = None


# ── Fact consolidation (LLM-based) ─────────────────────────

class ConsolidatedFact(BaseModel):
    category: ObservationCategory
    content: str
    date: str = ""
    valence: Literal["positive", "negative", "neutral", ""] = ""
    tags: list[str] = Field(default_factory=list)
    replaces_indices: list[int] = Field(default_factory=list)


class FactConsolidation(BaseModel):
    consolidated: list[ConsolidatedFact]


# ── Dream mode planning ─────────────────────────────────────

class DreamPlan(BaseModel):
    steps: list[int] = Field(description="Ordered list of step numbers to execute (1-10)")
    reasoning: str = Field(description="Brief explanation of why these steps were chosen")


class DreamValidation(BaseModel):
    approved: bool = Field(description="Whether the results look correct")
    issues: list[str] = Field(default_factory=list, description="List of issues found, if any")


# ── Dream mode deduplication ──────────────────────────────

class DedupVerdict(BaseModel):
    is_duplicate: bool
    confidence: float = Field(ge=0, le=1, default=0.5)
    reason: str = ""
