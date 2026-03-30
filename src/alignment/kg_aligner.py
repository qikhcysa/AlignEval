"""Knowledge graph alignment: match triples between source and learned KGs."""
from __future__ import annotations

import logging
from difflib import SequenceMatcher

from src.models import KnowledgeGraph, AlignmentResult, EvaluationMetrics

logger = logging.getLogger(__name__)


def _text_similarity(a: str, b: str) -> float:
    """Compute normalized string similarity using SequenceMatcher."""
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _triple_similarity(
    t1: tuple[str, str, str], t2: tuple[str, str, str], threshold: float
) -> float:
    """Compute similarity between two triples (head, relation, tail)."""
    head_sim = _text_similarity(t1[0], t2[0])
    rel_sim = _text_similarity(t1[1], t2[1])
    tail_sim = _text_similarity(t1[2], t2[2])
    # Weighted combination: entity names matter more than relation label
    score = 0.4 * head_sim + 0.2 * rel_sim + 0.4 * tail_sim
    return score


class KGAligner:
    """Align two knowledge graphs by matching their triples."""

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Args:
            similarity_threshold: Minimum similarity score to consider two triples a match.
        """
        self.similarity_threshold = similarity_threshold

    def align(
        self,
        source_kg: KnowledgeGraph,
        learned_kg: KnowledgeGraph,
    ) -> list[AlignmentResult]:
        """Align source KG triples against learned KG triples.

        Returns a list of AlignmentResult, one per source triple.
        """
        source_triples = [r.triple for r in source_kg.relations]
        learned_triples = [r.triple for r in learned_kg.relations]

        logger.info(
            "Aligning %d source triples against %d learned triples",
            len(source_triples),
            len(learned_triples),
        )

        results: list[AlignmentResult] = []

        for src_triple in source_triples:
            best_match: tuple[str, str, str] | None = None
            best_score = 0.0

            for lrn_triple in learned_triples:
                score = _triple_similarity(src_triple, lrn_triple, self.similarity_threshold)
                if score > best_score:
                    best_score = score
                    best_match = lrn_triple

            matched = best_score >= self.similarity_threshold
            results.append(
                AlignmentResult(
                    source_triple=src_triple,
                    matched=matched,
                    matched_triple=best_match if matched else None,
                    similarity=round(best_score, 4),
                )
            )

        return results

    def compute_metrics(
        self,
        source_kg: KnowledgeGraph,
        learned_kg: KnowledgeGraph,
    ) -> EvaluationMetrics:
        """Compute precision, recall, and F1 for the learned KG against the source KG."""
        alignment_details = self.align(source_kg, learned_kg)
        source_triples = [r.triple for r in source_kg.relations]
        learned_triples = [r.triple for r in learned_kg.relations]

        return EvaluationMetrics.compute(
            source_triples=source_triples,
            learned_triples=learned_triples,
            alignment_details=alignment_details,
        )
