"""Data models for AlignEval."""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


class KGSource(str, Enum):
    SOURCE = "source"
    LEARNED = "learned"


class ProbeLevel(str, Enum):
    FACTUAL = "factual"
    RELATIONAL = "relational"
    REVERSE = "reverse"


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    entity_type: str
    normalized: str = ""
    confidence: float = 1.0
    source: KGSource = KGSource.SOURCE

    def model_post_init(self, __context: Any) -> None:
        if not self.normalized:
            self.normalized = self.text.lower().strip()


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    head_id: str
    tail_id: str
    head_text: str
    tail_text: str
    relation_type: str
    confidence: float = 1.0
    source: KGSource = KGSource.SOURCE
    evidence: str = ""

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.head_text.lower(), self.relation_type.lower(), self.tail_text.lower())


class KnowledgeGraph(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source: KGSource
    entities: dict[str, Entity] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_entity(self, entity: Entity) -> Entity:
        key = entity.normalized
        if key not in self.entities:
            self.entities[key] = entity
        return self.entities[key]

    def add_relation(self, relation: Relation) -> None:
        triple = relation.triple
        existing = {r.triple for r in self.relations}
        if triple not in existing:
            self.relations.append(relation)

    def entity_count(self) -> int:
        return len(self.entities)

    def relation_count(self) -> int:
        return len(self.relations)

    def to_networkx(self):
        import networkx as nx
        G = nx.DiGraph()
        for key, entity in self.entities.items():
            G.add_node(
                entity.normalized,
                text=entity.text,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
            )
        for rel in self.relations:
            G.add_edge(
                rel.head_text.lower(),
                rel.tail_text.lower(),
                relation_type=rel.relation_type,
                confidence=rel.confidence,
                evidence=rel.evidence,
            )
        return G


class QAPair(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    context: str = ""
    domain: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProbePrompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    level: ProbeLevel
    prompt_text: str
    entity: str
    related_entity: str = ""
    expected_relation: str = ""


class ProbeResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: ProbePrompt
    response: str
    extracted_triples: list[tuple[str, str, str]] = Field(default_factory=list)
    model_name: str = ""
    latency_ms: float = 0.0


class AlignmentResult(BaseModel):
    source_triple: tuple[str, str, str]
    matched: bool
    matched_triple: tuple[str, str, str] | None = None
    similarity: float = 0.0


class EvaluationMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    correct_count: int
    total_learned: int
    total_source: int
    missing_triples: list[tuple[str, str, str]] = Field(default_factory=list)
    wrong_triples: list[tuple[str, str, str]] = Field(default_factory=list)
    alignment_details: list[AlignmentResult] = Field(default_factory=list)

    @classmethod
    def compute(
        cls,
        source_triples: list[tuple[str, str, str]],
        learned_triples: list[tuple[str, str, str]],
        alignment_details: list[AlignmentResult],
    ) -> "EvaluationMetrics":
        matched_source = {a.source_triple for a in alignment_details if a.matched}
        matched_learned = {a.matched_triple for a in alignment_details if a.matched and a.matched_triple}

        correct = len(matched_source)
        total_source = len(source_triples)
        total_learned = len(learned_triples)

        precision = correct / total_learned if total_learned > 0 else 0.0
        recall = correct / total_source if total_source > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        missing = [t for t in source_triples if t not in matched_source]
        wrong = [t for t in learned_triples if t not in matched_learned]

        return cls(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            correct_count=correct,
            total_learned=total_learned,
            total_source=total_source,
            missing_triples=missing,
            wrong_triples=wrong,
            alignment_details=alignment_details,
        )


class EvaluationSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_kg: KnowledgeGraph | None = None
    learned_kg: KnowledgeGraph | None = None
    probe_results: list[ProbeResult] = Field(default_factory=list)
    metrics: EvaluationMetrics | None = None
    status: str = "pending"
    domain: str = "general"
    model_name: str = ""
