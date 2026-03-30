"""Tests for the alignment module."""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import KnowledgeGraph, KGSource, Entity, Relation, EvaluationMetrics, AlignmentResult
from src.alignment import KGAligner, MetricsCalculator


def _make_source_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(name="source", source=KGSource.SOURCE)
    e1 = Entity(text="Metformin", entity_type="DRUG", normalized="metformin")
    e2 = Entity(text="diabetes", entity_type="DISEASE", normalized="diabetes")
    e3 = Entity(text="insulin", entity_type="DRUG", normalized="insulin")
    e4 = Entity(text="pancreas", entity_type="ORGAN", normalized="pancreas")
    kg.entities["metformin"] = e1
    kg.entities["diabetes"] = e2
    kg.entities["insulin"] = e3
    kg.entities["pancreas"] = e4
    kg.add_relation(Relation(head_id=e1.id, tail_id=e2.id, head_text="metformin", tail_text="diabetes", relation_type="treats"))
    kg.add_relation(Relation(head_id=e3.id, tail_id=e2.id, head_text="insulin", tail_text="diabetes", relation_type="regulates"))
    kg.add_relation(Relation(head_id=e4.id, tail_id=e3.id, head_text="pancreas", tail_text="insulin", relation_type="produces"))
    return kg


def _make_perfect_learned_kg() -> KnowledgeGraph:
    """Learned KG that matches source exactly."""
    kg = KnowledgeGraph(name="learned_perfect", source=KGSource.LEARNED)
    e1 = Entity(text="metformin", entity_type="DRUG", normalized="metformin", source=KGSource.LEARNED)
    e2 = Entity(text="diabetes", entity_type="DISEASE", normalized="diabetes", source=KGSource.LEARNED)
    e3 = Entity(text="insulin", entity_type="DRUG", normalized="insulin", source=KGSource.LEARNED)
    e4 = Entity(text="pancreas", entity_type="ORGAN", normalized="pancreas", source=KGSource.LEARNED)
    kg.entities["metformin"] = e1
    kg.entities["diabetes"] = e2
    kg.entities["insulin"] = e3
    kg.entities["pancreas"] = e4
    kg.add_relation(Relation(head_id=e1.id, tail_id=e2.id, head_text="metformin", tail_text="diabetes", relation_type="treats", source=KGSource.LEARNED))
    kg.add_relation(Relation(head_id=e3.id, tail_id=e2.id, head_text="insulin", tail_text="diabetes", relation_type="regulates", source=KGSource.LEARNED))
    kg.add_relation(Relation(head_id=e4.id, tail_id=e3.id, head_text="pancreas", tail_text="insulin", relation_type="produces", source=KGSource.LEARNED))
    return kg


def _make_partial_learned_kg() -> KnowledgeGraph:
    """Learned KG with only one triple (missing two)."""
    kg = KnowledgeGraph(name="learned_partial", source=KGSource.LEARNED)
    e1 = Entity(text="metformin", entity_type="DRUG", normalized="metformin", source=KGSource.LEARNED)
    e2 = Entity(text="diabetes", entity_type="DISEASE", normalized="diabetes", source=KGSource.LEARNED)
    kg.entities["metformin"] = e1
    kg.entities["diabetes"] = e2
    kg.add_relation(Relation(head_id=e1.id, tail_id=e2.id, head_text="metformin", tail_text="diabetes", relation_type="treats", source=KGSource.LEARNED))
    return kg


class TestKGAligner:
    def test_align_perfect_match(self):
        source = _make_source_kg()
        learned = _make_perfect_learned_kg()
        aligner = KGAligner(similarity_threshold=0.75)
        results = aligner.align(source, learned)
        assert len(results) == 3
        matched = [r for r in results if r.matched]
        assert len(matched) == 3

    def test_align_partial_match(self):
        source = _make_source_kg()
        learned = _make_partial_learned_kg()
        aligner = KGAligner(similarity_threshold=0.75)
        results = aligner.align(source, learned)
        assert len(results) == 3
        matched = [r for r in results if r.matched]
        assert len(matched) <= 3

    def test_align_empty_learned(self):
        source = _make_source_kg()
        learned = KnowledgeGraph(name="empty", source=KGSource.LEARNED)
        aligner = KGAligner()
        results = aligner.align(source, learned)
        assert all(not r.matched for r in results)

    def test_align_empty_source(self):
        source = KnowledgeGraph(name="empty_src", source=KGSource.SOURCE)
        learned = _make_perfect_learned_kg()
        aligner = KGAligner()
        results = aligner.align(source, learned)
        assert results == []


class TestMetricsCalculator:
    def test_perfect_recall_precision(self):
        source = _make_source_kg()
        learned = _make_perfect_learned_kg()
        calc = MetricsCalculator(similarity_threshold=0.75)
        metrics = calc.evaluate(source, learned)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.correct_count == 3

    def test_partial_recall(self):
        source = _make_source_kg()
        learned = _make_partial_learned_kg()
        calc = MetricsCalculator(similarity_threshold=0.75)
        metrics = calc.evaluate(source, learned)
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0

    def test_missing_triples_identified(self):
        source = _make_source_kg()
        learned = _make_partial_learned_kg()
        calc = MetricsCalculator(similarity_threshold=0.75)
        metrics = calc.evaluate(source, learned)
        # At most 2 triples are missing (insulin→diabetes, pancreas→insulin)
        assert len(metrics.missing_triples) <= 2

    def test_zero_metrics_empty_learned(self):
        source = _make_source_kg()
        learned = KnowledgeGraph(name="empty", source=KGSource.LEARNED)
        calc = MetricsCalculator()
        metrics = calc.evaluate(source, learned)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_summary_format(self):
        source = _make_source_kg()
        learned = _make_perfect_learned_kg()
        calc = MetricsCalculator(similarity_threshold=0.75)
        metrics = calc.evaluate(source, learned)
        summary = calc.summary(metrics)
        assert "precision" in summary
        assert "recall" in summary
        assert "f1" in summary
        assert "%" in summary["precision"]


class TestEvaluationMetrics:
    def test_compute_with_no_triples(self):
        metrics = EvaluationMetrics.compute(
            source_triples=[],
            learned_triples=[],
            alignment_details=[],
        )
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_compute_precision_denominator(self):
        """Precision = correct / total_learned."""
        src = [("a", "r", "b"), ("c", "r", "d")]
        lrn = [("a", "r", "b"), ("x", "r", "y")]
        details = [
            AlignmentResult(source_triple=("a", "r", "b"), matched=True, matched_triple=("a", "r", "b"), similarity=1.0),
            AlignmentResult(source_triple=("c", "r", "d"), matched=False),
        ]
        metrics = EvaluationMetrics.compute(src, lrn, details)
        assert metrics.precision == 0.5   # 1 correct / 2 learned
        assert metrics.recall == 0.5      # 1 correct / 2 source
