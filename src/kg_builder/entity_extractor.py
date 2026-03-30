"""Named Entity Recognition for knowledge graph construction."""
from __future__ import annotations

import re
import logging
from typing import Any

import spacy
from spacy.language import Language

from src.models import Entity, KGSource

logger = logging.getLogger(__name__)

# Map spaCy NER labels to domain-friendly type names
SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "FACILITY",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "DATE": "DATE",
    "TIME": "TIME",
    "PERCENT": "PERCENT",
    "MONEY": "MONEY",
    "QUANTITY": "QUANTITY",
    "ORDINAL": "ORDINAL",
    "CARDINAL": "CARDINAL",
    "NORP": "GROUP",
}

# Domain-specific entity patterns (regex-based)
DOMAIN_PATTERNS: list[dict[str, Any]] = [
    # Medical
    {"pattern": r"\b(?:COVID[-‐]?19|SARS[-‐]CoV[-‐]2|HIV|AIDS|cancer|tumor|diabetes|hypertension)\b",
     "label": "DISEASE"},
    {"pattern": r"\b(?:aspirin|ibuprofen|penicillin|amoxicillin|metformin|insulin)\b",
     "label": "DRUG", "flags": re.IGNORECASE},
    # Science / Tech
    {"pattern": r"\b(?:transformer|BERT|GPT|LLM|neural network|machine learning|deep learning)\b",
     "label": "AI_CONCEPT"},
    {"pattern": r"\b(?:Python|Java|C\+\+|JavaScript|TypeScript|Rust|Go)\b",
     "label": "PROGRAMMING_LANGUAGE"},
    # Legal / Finance
    {"pattern": r"\b(?:statute|regulation|act|law|policy|contract|patent)\b",
     "label": "LEGAL_CONCEPT", "flags": re.IGNORECASE},
    {"pattern": r"\b(?:GDP|inflation|interest rate|stock|bond|equity)\b",
     "label": "ECONOMIC_CONCEPT", "flags": re.IGNORECASE},
]

_NLP_CACHE: dict[str, Language] = {}


def _load_nlp(model_name: str = "en_core_web_sm") -> Language:
    if model_name not in _NLP_CACHE:
        try:
            _NLP_CACHE[model_name] = spacy.load(model_name)
            logger.info("Loaded spaCy model: %s", model_name)
        except OSError:
            logger.warning("spaCy model '%s' not found; falling back to blank model.", model_name)
            _NLP_CACHE[model_name] = spacy.blank("en")
    return _NLP_CACHE[model_name]


class EntityExtractor:
    """Extract named entities from text using spaCy NER + domain patterns."""

    def __init__(self, spacy_model: str = "en_core_web_sm", source: KGSource = KGSource.SOURCE):
        self.nlp = _load_nlp(spacy_model)
        self.source = source
        self._compiled_patterns = [
            (re.compile(p["pattern"], p.get("flags", 0)), p["label"])
            for p in DOMAIN_PATTERNS
        ]

    def extract(self, text: str, min_length: int = 2) -> list[Entity]:
        """Extract entities from a single text string."""
        entities: list[Entity] = []
        seen: set[str] = set()

        # spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            normalized = ent.text.lower().strip()
            if len(normalized) < min_length or normalized in seen:
                continue
            entity_type = SPACY_LABEL_MAP.get(ent.label_, ent.label_)
            entities.append(
                Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    normalized=normalized,
                    source=self.source,
                )
            )
            seen.add(normalized)

        # Domain-specific patterns
        for pattern, label in self._compiled_patterns:
            for match in pattern.finditer(text):
                normalized = match.group().lower().strip()
                if len(normalized) < min_length or normalized in seen:
                    continue
                entities.append(
                    Entity(
                        text=match.group(),
                        entity_type=label,
                        normalized=normalized,
                        source=self.source,
                    )
                )
                seen.add(normalized)

        return entities

    def extract_batch(self, texts: list[str], min_length: int = 2) -> list[list[Entity]]:
        """Extract entities from a list of texts."""
        return [self.extract(t, min_length=min_length) for t in texts]

    def extract_unique(self, texts: list[str]) -> list[Entity]:
        """Extract unique entities across all texts."""
        seen: set[str] = set()
        unique: list[Entity] = []
        for text in texts:
            for entity in self.extract(text):
                if entity.normalized not in seen:
                    unique.append(entity)
                    seen.add(entity.normalized)
        return unique
