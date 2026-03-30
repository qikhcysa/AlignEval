"""Knowledge graph constructor: orchestrates NER + RE to build KG from Q&A data."""
from __future__ import annotations

import logging
from typing import Any

from src.models import KnowledgeGraph, KGSource, QAPair, Entity, Relation
from src.kg_builder.entity_extractor import EntityExtractor
from src.kg_builder.relation_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class KGConstructor:
    """Build a KnowledgeGraph from a list of QAPairs using NER + RE."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        source: KGSource = KGSource.SOURCE,
        min_entity_length: int = 2,
    ):
        self.entity_extractor = EntityExtractor(spacy_model=spacy_model, source=source)
        self.relation_extractor = RelationExtractor(spacy_model=spacy_model, source=source)
        self.source = source
        self.min_entity_length = min_entity_length

    def build_from_qa_pairs(
        self, qa_pairs: list[QAPair], name: str = "knowledge_graph"
    ) -> KnowledgeGraph:
        """Build KG from a list of QAPair objects."""
        kg = KnowledgeGraph(name=name, source=self.source)
        kg.metadata["qa_count"] = len(qa_pairs)

        for qa in qa_pairs:
            texts = [qa.question, qa.answer]
            if qa.context:
                texts.append(qa.context)

            full_text = " ".join(texts)
            entities = self.entity_extractor.extract(full_text, min_length=self.min_entity_length)

            for entity in entities:
                kg.add_entity(entity)

            relations = self.relation_extractor.extract_from_text(
                full_text, list(kg.entities.values())
            )
            for relation in relations:
                relation.source = self.source
                kg.add_relation(relation)

        logger.info(
            "Built KG '%s': %d entities, %d relations from %d QA pairs",
            name,
            kg.entity_count(),
            kg.relation_count(),
            len(qa_pairs),
        )
        return kg

    def build_from_texts(
        self, texts: list[str], name: str = "knowledge_graph"
    ) -> KnowledgeGraph:
        """Build KG from a list of plain text strings."""
        qa_pairs = [
            QAPair(question="", answer=t) for t in texts
        ]
        return self.build_from_qa_pairs(qa_pairs, name=name)

    def build_from_dicts(
        self, records: list[dict[str, Any]], name: str = "knowledge_graph"
    ) -> KnowledgeGraph:
        """Build KG from a list of dicts with 'question'/'answer' keys."""
        qa_pairs = []
        for rec in records:
            qa_pairs.append(
                QAPair(
                    question=rec.get("question", ""),
                    answer=rec.get("answer", ""),
                    context=rec.get("context", ""),
                    domain=rec.get("domain", "general"),
                )
            )
        return self.build_from_qa_pairs(qa_pairs, name=name)
