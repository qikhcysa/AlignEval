"""Relation extraction for knowledge graph construction."""
from __future__ import annotations

import logging
import re
from typing import Any

import spacy
from spacy.language import Language

from src.models import Entity, Relation, KGSource
from src.kg_builder.entity_extractor import _load_nlp

logger = logging.getLogger(__name__)

# Syntactic dependency relation patterns
# Each rule checks the dependency path between two entities in a sentence.
RELATION_TEMPLATES: list[dict[str, Any]] = [
    # X is-a / is-type-of Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:is|are|was|were)\s+(?:a|an|the|one of)?\s+(?P<tail>.+)",
     "relation": "is_a"},
    # X causes/leads to/results in Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:causes?|leads? to|results? in|triggers?|induces?)\s+(?P<tail>.+)",
     "relation": "causes"},
    # X treats/cures Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:treats?|cures?|heals?|manages?|addresses?)\s+(?P<tail>.+)",
     "relation": "treats"},
    # X contains/includes Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:contains?|includes?|comprises?|consists? of)\s+(?P<tail>.+)",
     "relation": "contains"},
    # X uses/utilizes Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:uses?|utilizes?|employs?|applies?)\s+(?P<tail>.+)",
     "relation": "uses"},
    # X belongs to Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:belongs? to|is part of|is a type of)\s+(?P<tail>.+)",
     "relation": "belongs_to"},
    # X produces/generates Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:produces?|generates?|creates?|yields?)\s+(?P<tail>.+)",
     "relation": "produces"},
    # X inhibits/blocks Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:inhibits?|blocks?|prevents?|suppresses?)\s+(?P<tail>.+)",
     "relation": "inhibits"},
    # X interacts with Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:interacts? with|reacts? with|binds? to)\s+(?P<tail>.+)",
     "relation": "interacts_with"},
    # X is associated with Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:is associated with|correlates? with|is related to)\s+(?P<tail>.+)",
     "relation": "associated_with"},
    # X defined as Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:is defined as|refers? to|means?)\s+(?P<tail>.+)",
     "relation": "defined_as"},
    # X has property Y
    {"pattern": r"\b(?P<head>.+?)\s+(?:has|have|had)\s+(?:a|an|the)?\s+(?P<tail>.+)",
     "relation": "has"},
]

_COMPILED_TEMPLATES: list[tuple[re.Pattern, str]] = [
    (re.compile(t["pattern"], re.IGNORECASE), t["relation"])
    for t in RELATION_TEMPLATES
]


class RelationExtractor:
    """Extract relations between entities using dependency parsing + pattern matching."""

    def __init__(self, spacy_model: str = "en_core_web_sm", source: KGSource = KGSource.SOURCE):
        self.nlp = _load_nlp(spacy_model)
        self.source = source

    def extract_from_sentence(
        self,
        sentence: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """Extract relations from a sentence given a list of candidate entities."""
        relations: list[Relation] = []
        entity_map = {e.normalized: e for e in entities}
        entity_texts = sorted(entity_map.keys(), key=len, reverse=True)

        if len(entity_texts) < 2:
            return relations

        # Try dependency-based extraction with spaCy
        dep_relations = self._dep_extract(sentence, entity_map)
        relations.extend(dep_relations)

        # Try pattern-based extraction
        pat_relations = self._pattern_extract(sentence, entity_map)
        relations.extend(pat_relations)

        # Deduplicate
        seen_triples: set[tuple[str, str, str]] = set()
        unique: list[Relation] = []
        for r in relations:
            t = r.triple
            if t not in seen_triples:
                seen_triples.add(t)
                unique.append(r)

        return unique

    def _dep_extract(self, sentence: str, entity_map: dict[str, Entity]) -> list[Relation]:
        """Use spaCy dependency parse to extract subject-verb-object relations."""
        relations: list[Relation] = []
        doc = self.nlp(sentence)

        # Build a token -> entity lookup
        def find_entity_for_span(span_text: str) -> Entity | None:
            norm = span_text.lower().strip()
            if norm in entity_map:
                return entity_map[norm]
            # Partial match
            for key, ent in entity_map.items():
                if key in norm or norm in key:
                    return ent
            return None

        for token in doc:
            if token.dep_ in ("ROOT", "relcl") and token.pos_ in ("VERB", "AUX"):
                subj_tokens = [t for t in token.children if t.dep_ in ("nsubj", "nsubjpass")]
                obj_tokens = [t for t in token.children if t.dep_ in ("dobj", "pobj", "attr", "acomp")]

                for subj in subj_tokens:
                    subj_span = self._expand_span(subj, doc)
                    for obj in obj_tokens:
                        obj_span = self._expand_span(obj, doc)
                        head_ent = find_entity_for_span(subj_span)
                        tail_ent = find_entity_for_span(obj_span)
                        if head_ent and tail_ent and head_ent.normalized != tail_ent.normalized:
                            verb_lemma = token.lemma_.lower()
                            relations.append(
                                Relation(
                                    head_id=head_ent.id,
                                    tail_id=tail_ent.id,
                                    head_text=head_ent.text,
                                    tail_text=tail_ent.text,
                                    relation_type=verb_lemma,
                                    source=self.source,
                                    evidence=sentence,
                                )
                            )
        return relations

    def _pattern_extract(self, sentence: str, entity_map: dict[str, Entity]) -> list[Relation]:
        """Use regex patterns to extract relations."""
        relations: list[Relation] = []
        entity_keys = sorted(entity_map.keys(), key=len, reverse=True)

        for pattern, relation_type in _COMPILED_TEMPLATES:
            match = pattern.search(sentence)
            if not match:
                continue
            try:
                head_text = match.group("head").strip()
                tail_text = match.group("tail").strip()
            except IndexError:
                continue

            # Find best matching entities
            head_ent = self._find_best_entity(head_text, entity_map)
            tail_ent = self._find_best_entity(tail_text, entity_map)

            if head_ent and tail_ent and head_ent.normalized != tail_ent.normalized:
                relations.append(
                    Relation(
                        head_id=head_ent.id,
                        tail_id=tail_ent.id,
                        head_text=head_ent.text,
                        tail_text=tail_ent.text,
                        relation_type=relation_type,
                        source=self.source,
                        evidence=sentence,
                    )
                )

        return relations

    @staticmethod
    def _expand_span(token, doc) -> str:
        """Expand a token to its full noun phrase."""
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk.text
        return token.text

    @staticmethod
    def _find_best_entity(text: str, entity_map: dict[str, Entity]) -> Entity | None:
        """Find the entity that best matches the given text."""
        norm = text.lower().strip()
        if norm in entity_map:
            return entity_map[norm]
        best_match: Entity | None = None
        best_overlap = 0
        for key, ent in entity_map.items():
            if key in norm:
                if len(key) > best_overlap:
                    best_overlap = len(key)
                    best_match = ent
            elif norm in key:
                if len(norm) > best_overlap:
                    best_overlap = len(norm)
                    best_match = ent
        return best_match

    def extract_from_text(self, text: str, entities: list[Entity]) -> list[Relation]:
        """Extract relations from full text by splitting into sentences."""
        doc = self.nlp(text)
        all_relations: list[Relation] = []
        for sent in doc.sents:
            rels = self.extract_from_sentence(sent.text, entities)
            all_relations.extend(rels)

        # Deduplicate across sentences
        seen: set[tuple[str, str, str]] = set()
        unique: list[Relation] = []
        for r in all_relations:
            t = r.triple
            if t not in seen:
                seen.add(t)
                unique.append(r)

        return unique
