"""Pydantic v2 models for memory-ai. All structured LLM outputs and data models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ── Closed lists (from spec §4) ──────────────────────────────

ObservationCategory = Literal[
    "fait", "preference", "diagnostic", "traitement", "progression",
    "technique", "vigilance", "decision", "emotion",
    "relation_interpersonnelle", "competence", "projet", "contexte", "regle",
]

EntityType = Literal[
    "personne", "sante", "travail", "projet", "interet",
    "lieu", "animal", "organisation",
]

RelationType = Literal[
    "affecte", "ameliore", "aggrave", "necessite", "lie_a",
    "vit_avec", "travaille_a", "parent_de", "ami_de", "utilise",
    "fait_partie_de", "contraste_avec", "precede",
]


# ── Step 1: Raw extraction output ────────────────────────────

class RawObservation(BaseModel):
    category: ObservationCategory
    content: str
    importance: float = Field(ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


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


class GraphRelation(BaseModel):
    from_entity: str = Field(alias="from", serialization_alias="from")
    to_entity: str = Field(alias="to", serialization_alias="to")
    type: RelationType

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
