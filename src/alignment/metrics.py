"""Metrics calculation utilities."""
from __future__ import annotations

from src.models import EvaluationMetrics, KnowledgeGraph
from src.alignment.kg_aligner import KGAligner


class MetricsCalculator:
    """High-level wrapper for evaluating a learned KG against a source KG."""

    def __init__(self, similarity_threshold: float = 0.75):
        self.aligner = KGAligner(similarity_threshold=similarity_threshold)

    def evaluate(self, source_kg: KnowledgeGraph, learned_kg: KnowledgeGraph) -> EvaluationMetrics:
        """Run full evaluation and return metrics."""
        return self.aligner.compute_metrics(source_kg, learned_kg)

    def summary(self, metrics: EvaluationMetrics) -> dict:
        """Return a human-readable summary dict."""
        return {
            "precision": f"{metrics.precision:.2%}",
            "recall": f"{metrics.recall:.2%}",
            "f1": f"{metrics.f1:.2%}",
            "correct_triples": metrics.correct_count,
            "total_source_triples": metrics.total_source,
            "total_learned_triples": metrics.total_learned,
            "missing_count": len(metrics.missing_triples),
            "wrong_count": len(metrics.wrong_triples),
        }
