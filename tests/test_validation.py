"""Tests for the fine-tuning validation module."""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import KnowledgeGraph, KGSource, Entity, Relation, ProbePrompt, ProbeLevel
from src.probing.model_prober import ModelProber
from src.validation import FineTuningValidator, ValidationReport


def _make_source_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(name="source", source=KGSource.SOURCE)
    e1 = Entity(text="Metformin", entity_type="DRUG", normalized="metformin")
    e2 = Entity(text="diabetes", entity_type="DISEASE", normalized="diabetes")
    e3 = Entity(text="insulin", entity_type="DRUG", normalized="insulin")
    kg.entities["metformin"] = e1
    kg.entities["diabetes"] = e2
    kg.entities["insulin"] = e3
    kg.add_relation(Relation(
        head_id=e1.id, tail_id=e2.id,
        head_text="Metformin", tail_text="diabetes",
        relation_type="treats",
    ))
    kg.add_relation(Relation(
        head_id=e3.id, tail_id=e2.id,
        head_text="insulin", tail_text="diabetes",
        relation_type="regulates",
    ))
    return kg


class TestModelProber:
    def test_mock_mode_explicit(self):
        prober = ModelProber(mock_mode=True)
        assert prober.mock_mode is True

    def test_no_model_forces_mock(self):
        """No model_name_or_path and no pre-loaded model → mock mode."""
        prober = ModelProber()
        assert prober.mock_mode is True

    def test_query_returns_probe_result(self):
        prober = ModelProber(mock_mode=True)
        prompt = ProbePrompt(
            level=ProbeLevel.FACTUAL,
            prompt_text="What is diabetes?",
            entity="diabetes",
        )
        result = prober.query(prompt)
        assert result.response
        assert result.prompt == prompt
        assert "mock" in result.model_name

    def test_query_factual_level(self):
        prober = ModelProber(mock_mode=True)
        prompt = ProbePrompt(
            level=ProbeLevel.FACTUAL,
            prompt_text="What is insulin?",
            entity="insulin",
        )
        result = prober.query(prompt)
        assert result.response
        assert result.latency_ms >= 0

    def test_query_relational_level(self):
        prober = ModelProber(mock_mode=True)
        prompt = ProbePrompt(
            level=ProbeLevel.RELATIONAL,
            prompt_text="How is metformin related to diabetes?",
            entity="metformin",
            related_entity="diabetes",
            expected_relation="treats",
        )
        result = prober.query(prompt)
        assert result.response
        assert result.prompt.level == ProbeLevel.RELATIONAL

    def test_query_reverse_level(self):
        prober = ModelProber(mock_mode=True)
        prompt = ProbePrompt(
            level=ProbeLevel.REVERSE,
            prompt_text="Why does metformin treat diabetes?",
            entity="metformin",
            related_entity="diabetes",
            expected_relation="treats",
        )
        result = prober.query(prompt)
        assert result.response

    def test_batch_query(self):
        prober = ModelProber(mock_mode=True)
        prompts = [
            ProbePrompt(
                level=ProbeLevel.FACTUAL,
                prompt_text=f"What is entity{i}?",
                entity=f"entity{i}",
            )
            for i in range(5)
        ]
        results = prober.query_batch(prompts)
        assert len(results) == 5
        for r in results:
            assert r.response

    def test_model_name_in_result(self):
        prober = ModelProber(mock_mode=True)
        prompt = ProbePrompt(
            level=ProbeLevel.FACTUAL,
            prompt_text="What is insulin?",
            entity="insulin",
        )
        result = prober.query(prompt)
        assert result.model_name  # not empty


class TestFineTuningValidator:
    def test_validate_produces_report(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator(domain="biomedical")
        probers = {
            "full":    ModelProber(mock_mode=True),
            "half":    ModelProber(mock_mode=True),
            "control": ModelProber(mock_mode=True),
        }
        report = validator.validate(source_kg, probers, experiment_name="test_exp")
        assert report.experiment_name == "test_exp"
        assert set(report.model_metrics.keys()) == {"full", "half", "control"}
        for label, metrics in report.model_metrics.items():
            assert 0.0 <= metrics.f1 <= 1.0

    def test_validate_metrics_are_valid(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator(domain="biomedical")
        report = validator.validate(
            source_kg,
            {"model_a": ModelProber(mock_mode=True)},
        )
        m = report.model_metrics["model_a"]
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert m.total_source == source_kg.relation_count()

    def test_f1_scores_returns_dict(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator()
        probers = {
            "model_a": ModelProber(mock_mode=True),
            "model_b": ModelProber(mock_mode=True),
        }
        report = validator.validate(source_kg, probers)
        scores = report.f1_scores()
        assert "model_a" in scores
        assert "model_b" in scores
        for v in scores.values():
            assert isinstance(v, float)

    def test_summary_format(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator()
        report = validator.validate(
            source_kg,
            {"full": ModelProber(mock_mode=True)},
            experiment_name="summary_test",
        )
        s = report.summary()
        assert s["experiment"] == "summary_test"
        assert "full" in s["models"]
        row = s["models"]["full"]
        assert "f1" in row
        assert "precision" in row
        assert "recall" in row
        assert "%" in row["f1"]

    def test_is_monotonic_with_equal_mock_scores(self):
        """Mock probers produce equal scores; non-strict ≥ must hold."""
        source_kg = _make_source_kg()
        validator = FineTuningValidator()
        probers = {
            "full":    ModelProber(mock_mode=True),
            "control": ModelProber(mock_mode=True),
        }
        report = validator.validate(source_kg, probers)
        # Equal scores still satisfy the non-strictly-decreasing condition.
        assert report.is_monotonic(["full", "control"])

    def test_is_monotonic_ignores_missing_labels(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator()
        report = validator.validate(
            source_kg,
            {"full": ModelProber(mock_mode=True)},
        )
        # Labels not in the report are silently skipped.
        assert report.is_monotonic(["full", "nonexistent"])

    def test_empty_model_probers(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator()
        report = validator.validate(source_kg, {})
        assert report.model_metrics == {}
        assert report.f1_scores() == {}

    def test_default_experiment_name(self):
        source_kg = _make_source_kg()
        validator = FineTuningValidator()
        report = validator.validate(source_kg, {})
        assert report.experiment_name == "finetuning_validation"
