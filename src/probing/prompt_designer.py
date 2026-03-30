"""Multi-level prompt designer for knowledge probing."""
from __future__ import annotations

import random
from src.models import KnowledgeGraph, ProbePrompt, ProbeLevel, Entity, Relation

# Level 1: Factual prompts – probe basic entity knowledge
FACTUAL_TEMPLATES = [
    "What is {entity}?",
    "Please describe {entity} in detail.",
    "What are the key properties or characteristics of {entity}?",
    "Can you explain what {entity} means?",
    "Define {entity} in the context of {domain}.",
]

# Level 2: Relational prompts – probe relationships between entities
RELATIONAL_TEMPLATES = [
    "How is {entity} related to {related_entity}?",
    "What is the relationship between {entity} and {related_entity}?",
    "Does {entity} have any effect on {related_entity}? If so, describe it.",
    "In the context of {domain}, how do {entity} and {related_entity} interact?",
    "Explain the connection between {entity} and {related_entity}.",
]

# Level 3: Reverse reasoning prompts – probe causal / inferential understanding
REVERSE_TEMPLATES = [
    "If {entity} {relation} {related_entity}, what are the underlying mechanisms or reasons?",
    "Given that {entity} and {related_entity} are related through '{relation}', can you explain why?",
    "What would happen to {related_entity} if {entity} was absent or removed?",
    "Reason backwards: given that {related_entity} is affected, what role does {entity} play?",
    "What preconditions are required for {entity} to {relation} {related_entity}?",
]


class PromptDesigner:
    """Design multi-level probing prompts for a knowledge graph."""

    def __init__(self, domain: str = "general", seed: int | None = None):
        self.domain = domain
        self._rng = random.Random(seed)

    def design_factual_prompts(self, kg: KnowledgeGraph, max_entities: int = 50) -> list[ProbePrompt]:
        """Level 1: Generate factual prompts for entities in the KG."""
        entities = list(kg.entities.values())[:max_entities]
        prompts: list[ProbePrompt] = []
        for entity in entities:
            template = self._rng.choice(FACTUAL_TEMPLATES)
            text = template.format(entity=entity.text, domain=self.domain)
            prompts.append(
                ProbePrompt(
                    level=ProbeLevel.FACTUAL,
                    prompt_text=text,
                    entity=entity.text,
                )
            )
        return prompts

    def design_relational_prompts(self, kg: KnowledgeGraph, max_relations: int = 50) -> list[ProbePrompt]:
        """Level 2: Generate relational prompts for entity pairs in the KG."""
        relations = kg.relations[:max_relations]
        prompts: list[ProbePrompt] = []
        for rel in relations:
            template = self._rng.choice(RELATIONAL_TEMPLATES)
            text = template.format(
                entity=rel.head_text,
                related_entity=rel.tail_text,
                domain=self.domain,
            )
            prompts.append(
                ProbePrompt(
                    level=ProbeLevel.RELATIONAL,
                    prompt_text=text,
                    entity=rel.head_text,
                    related_entity=rel.tail_text,
                    expected_relation=rel.relation_type,
                )
            )
        return prompts

    def design_reverse_prompts(self, kg: KnowledgeGraph, max_relations: int = 30) -> list[ProbePrompt]:
        """Level 3: Generate reverse reasoning prompts."""
        relations = kg.relations[:max_relations]
        prompts: list[ProbePrompt] = []
        for rel in relations:
            template = self._rng.choice(REVERSE_TEMPLATES)
            text = template.format(
                entity=rel.head_text,
                related_entity=rel.tail_text,
                relation=rel.relation_type.replace("_", " "),
                domain=self.domain,
            )
            prompts.append(
                ProbePrompt(
                    level=ProbeLevel.REVERSE,
                    prompt_text=text,
                    entity=rel.head_text,
                    related_entity=rel.tail_text,
                    expected_relation=rel.relation_type,
                )
            )
        return prompts

    def design_all_prompts(
        self,
        kg: KnowledgeGraph,
        max_entities: int = 50,
        max_relations: int = 50,
    ) -> list[ProbePrompt]:
        """Design prompts for all three levels."""
        prompts = []
        prompts.extend(self.design_factual_prompts(kg, max_entities=max_entities))
        prompts.extend(self.design_relational_prompts(kg, max_relations=max_relations))
        prompts.extend(self.design_reverse_prompts(kg, max_relations=max_relations // 2))
        return prompts
