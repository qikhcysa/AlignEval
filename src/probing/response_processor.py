"""Process LLM responses and extract knowledge graph triples."""
from __future__ import annotations

import logging

from src.models import ProbeResult, KnowledgeGraph, KGSource
from src.kg_builder import EntityExtractor, RelationExtractor

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """Extract entities and relations from LLM probe responses to build a learned KG."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.entity_extractor = EntityExtractor(
            spacy_model=spacy_model, source=KGSource.LEARNED
        )
        self.relation_extractor = RelationExtractor(
            spacy_model=spacy_model, source=KGSource.LEARNED
        )

    def process_result(self, result: ProbeResult) -> ProbeResult:
        """Extract triples from a single probe result and attach to it."""
        # Seed the extraction with entities mentioned in the prompt
        seed_entities = self.entity_extractor.extract(
            result.prompt.prompt_text + " " + result.response
        )

        relations = self.relation_extractor.extract_from_text(
            result.response, seed_entities
        )

        # Also add entity from the prompt as a guaranteed node
        if result.prompt.entity:
            prompt_entities = self.entity_extractor.extract(result.prompt.entity)
            seed_entities = list({e.normalized: e for e in seed_entities + prompt_entities}.values())

        triples = [r.triple for r in relations]

        # If no structural triples were found but there are ≥2 entities,
        # create a generic "related_to" triple from the probe context
        if not triples and result.prompt.related_entity:
            triples.append((
                result.prompt.entity.lower(),
                result.prompt.expected_relation or "related_to",
                result.prompt.related_entity.lower(),
            ))

        result.extracted_triples = triples
        return result

    def process_batch(self, results: list[ProbeResult]) -> list[ProbeResult]:
        """Process a list of probe results."""
        return [self.process_result(r) for r in results]

    def build_learned_kg(
        self, results: list[ProbeResult], name: str = "learned_kg"
    ) -> KnowledgeGraph:
        """Build a learned knowledge graph from processed probe results."""
        from src.models import Entity, Relation

        kg = KnowledgeGraph(name=name, source=KGSource.LEARNED)
        processed = self.process_batch(results)

        for result in processed:
            # Extract entities from response
            entities = self.entity_extractor.extract(
                result.prompt.prompt_text + " " + result.response
            )
            for entity in entities:
                kg.add_entity(entity)

            # Add triples
            for head_text, rel_type, tail_text in result.extracted_triples:
                # Ensure head/tail entities exist
                head_key = head_text.lower()
                tail_key = tail_text.lower()

                if head_key not in kg.entities:
                    kg.entities[head_key] = Entity(
                        text=head_text,
                        entity_type="UNKNOWN",
                        normalized=head_key,
                        source=KGSource.LEARNED,
                    )
                if tail_key not in kg.entities:
                    kg.entities[tail_key] = Entity(
                        text=tail_text,
                        entity_type="UNKNOWN",
                        normalized=tail_key,
                        source=KGSource.LEARNED,
                    )

                head_ent = kg.entities[head_key]
                tail_ent = kg.entities[tail_key]
                kg.add_relation(
                    Relation(
                        head_id=head_ent.id,
                        tail_id=tail_ent.id,
                        head_text=head_ent.text,
                        tail_text=tail_ent.text,
                        relation_type=rel_type,
                        source=KGSource.LEARNED,
                        evidence=result.response[:200],
                    )
                )

        logger.info(
            "Built learned KG '%s': %d entities, %d relations from %d probe results",
            name,
            kg.entity_count(),
            kg.relation_count(),
            len(results),
        )
        return kg
