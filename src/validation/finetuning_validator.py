"""Fine-tuning validation orchestrator.

Provides :class:`FineTuningValidator`, which probes multiple models against a
shared source KG and checks whether AlignEval scores reproduce the expected
knowledge-coverage ranking — thereby validating the evaluation system itself.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.models import KnowledgeGraph, EvaluationMetrics
from src.probing.model_prober import ModelProber
from src.probing.prompt_designer import PromptDesigner
from src.probing.response_processor import ResponseProcessor
from src.alignment.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ValidationReport(BaseModel):
    """Aggregated results of a fine-tuning validation experiment.

    Attributes
    ----------
    experiment_name:
        Human-readable label for the experiment.
    model_metrics:
        Mapping from model label (e.g. ``"full"``, ``"half"``,
        ``"control"``) to its :class:`~src.models.EvaluationMetrics`.
    """

    experiment_name: str
    model_metrics: dict[str, EvaluationMetrics] = Field(default_factory=dict)

    def f1_scores(self) -> dict[str, float]:
        """Return ``{label: f1}`` for every evaluated model."""
        return {label: m.f1 for label, m in self.model_metrics.items()}

    def is_monotonic(self, ordered_labels: list[str]) -> bool:
        """Return ``True`` if F1 scores are non-increasing along *ordered_labels*.

        Use this to assert that a model trained on more data receives a higher
        AlignEval score than one trained on less data::

            assert report.is_monotonic(["full", "half", "control"])
        """
        scores = [
            self.model_metrics[label].f1
            for label in ordered_labels
            if label in self.model_metrics
        ]
        return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        rows: dict[str, Any] = {}
        for label, m in self.model_metrics.items():
            rows[label] = {
                "precision": f"{m.precision:.2%}",
                "recall": f"{m.recall:.2%}",
                "f1": f"{m.f1:.2%}",
                "correct_triples": m.correct_count,
                "total_source_triples": m.total_source,
                "total_learned_triples": m.total_learned,
            }
        return {"experiment": self.experiment_name, "models": rows}


class FineTuningValidator:
    """Orchestrate a controlled fine-tuning validation experiment.

    The validator probes each model in *model_probers* against the same
    *source_kg*, collects AlignEval scores, and packages them in a
    :class:`ValidationReport`.

    Typical usage::

        from src.probing import ModelProber
        from src.validation import FineTuningValidator

        validator = FineTuningValidator(domain="biomedical")
        report = validator.validate(
            source_kg=source_kg,
            model_probers={
                "full":    ModelProber(model_name_or_path="./checkpoints/full"),
                "half":    ModelProber(model_name_or_path="./checkpoints/half"),
                "control": ModelProber(mock_mode=True),
            },
        )
        print(report.summary())
        assert report.is_monotonic(["full", "half", "control"])

    Parameters
    ----------
    domain:
        Domain label forwarded to :class:`~src.probing.PromptDesigner`.
    similarity_threshold:
        Forwarded to :class:`~src.alignment.metrics.MetricsCalculator`.
    max_entities:
        Maximum number of entities probed per model (controls prompt volume).
    max_relations:
        Maximum number of relations probed per model.
    """

    def __init__(
        self,
        domain: str = "general",
        similarity_threshold: float = 0.75,
        max_entities: int = 50,
        max_relations: int = 50,
    ):
        self.domain = domain
        self.similarity_threshold = similarity_threshold
        self.max_entities = max_entities
        self.max_relations = max_relations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_model(
        self,
        prober: ModelProber,
        source_kg: KnowledgeGraph,
        label: str,
    ) -> KnowledgeGraph:
        """Probe *prober* against *source_kg* and return a learned KG."""
        designer = PromptDesigner(domain=self.domain)
        prompts = designer.design_all_prompts(
            source_kg,
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )
        results = prober.query_batch(prompts)
        processor = ResponseProcessor()
        learned_kg = processor.build_learned_kg(results, name=f"learned_{label}")
        logger.info(
            "Probed model '%s': %d prompts → %d entities, %d relations in learned KG",
            label,
            len(prompts),
            learned_kg.entity_count(),
            learned_kg.relation_count(),
        )
        return learned_kg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        source_kg: KnowledgeGraph,
        model_probers: dict[str, ModelProber],
        experiment_name: str = "finetuning_validation",
    ) -> ValidationReport:
        """Run the full validation experiment.

        Parameters
        ----------
        source_kg:
            The ground-truth knowledge graph built from the test Q&A data.
        model_probers:
            Mapping from experiment label to its :class:`~src.probing.ModelProber`.
        experiment_name:
            Human-readable name stored in the returned :class:`ValidationReport`.

        Returns
        -------
        ValidationReport
            Contains per-model :class:`~src.models.EvaluationMetrics` and
            convenience helpers for asserting score monotonicity.
        """
        calculator = MetricsCalculator(similarity_threshold=self.similarity_threshold)
        report = ValidationReport(experiment_name=experiment_name)

        for label, prober in model_probers.items():
            learned_kg = self._probe_model(prober, source_kg, label)
            metrics = calculator.evaluate(source_kg, learned_kg)
            report.model_metrics[label] = metrics
            logger.info(
                "Experiment '%s' — model '%s': P=%.2f R=%.2f F1=%.2f",
                experiment_name,
                label,
                metrics.precision,
                metrics.recall,
                metrics.f1,
            )

        return report
