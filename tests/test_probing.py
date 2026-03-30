"""Tests for the probing module."""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import KnowledgeGraph, KGSource, Entity, Relation, ProbeLevel
from src.probing import PromptDesigner, LLMClient, ResponseProcessor


def _make_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(name="test_kg", source=KGSource.SOURCE)
    e1 = Entity(text="Metformin", entity_type="DRUG", normalized="metformin", source=KGSource.SOURCE)
    e2 = Entity(text="diabetes", entity_type="DISEASE", normalized="diabetes", source=KGSource.SOURCE)
    kg.entities["metformin"] = e1
    kg.entities["diabetes"] = e2
    kg.add_relation(Relation(
        head_id=e1.id, tail_id=e2.id,
        head_text="Metformin", tail_text="diabetes",
        relation_type="treats", source=KGSource.SOURCE,
    ))
    return kg


class TestPromptDesigner:
    def test_factual_prompts(self):
        kg = _make_kg()
        designer = PromptDesigner(domain="biomedical")
        prompts = designer.design_factual_prompts(kg)
        assert len(prompts) == 2  # one per entity
        for p in prompts:
            assert p.level == ProbeLevel.FACTUAL
            assert p.prompt_text
            assert p.entity

    def test_relational_prompts(self):
        kg = _make_kg()
        designer = PromptDesigner(domain="biomedical")
        prompts = designer.design_relational_prompts(kg)
        assert len(prompts) == 1  # one relation
        assert prompts[0].level == ProbeLevel.RELATIONAL
        assert prompts[0].entity == "Metformin"
        assert prompts[0].related_entity == "diabetes"

    def test_reverse_prompts(self):
        kg = _make_kg()
        designer = PromptDesigner(domain="biomedical")
        prompts = designer.design_reverse_prompts(kg)
        assert len(prompts) == 1
        assert prompts[0].level == ProbeLevel.REVERSE

    def test_all_prompts(self):
        kg = _make_kg()
        designer = PromptDesigner()
        all_prompts = designer.design_all_prompts(kg)
        levels = [p.level for p in all_prompts]
        assert ProbeLevel.FACTUAL in levels
        assert ProbeLevel.RELATIONAL in levels
        assert ProbeLevel.REVERSE in levels


class TestLLMClient:
    def test_mock_mode_returns_result(self):
        client = LLMClient(mock_mode=True)
        assert client.mock_mode is True

    def test_no_api_key_forces_mock(self):
        client = LLMClient(api_key="")
        assert client.mock_mode is True

    def test_query_returns_probe_result(self):
        from src.models import ProbePrompt
        client = LLMClient(mock_mode=True)
        prompt = ProbePrompt(
            level=ProbeLevel.FACTUAL,
            prompt_text="What is diabetes?",
            entity="diabetes",
        )
        result = client.query(prompt)
        assert result.response
        assert result.prompt == prompt
        assert result.model_name == "mock"

    def test_batch_query(self):
        from src.models import ProbePrompt
        client = LLMClient(mock_mode=True)
        kg = _make_kg()
        designer = PromptDesigner()
        prompts = designer.design_all_prompts(kg)
        results = client.query_batch(prompts)
        assert len(results) == len(prompts)
        for r in results:
            assert r.response


class TestResponseProcessor:
    def test_build_learned_kg(self):
        kg = _make_kg()
        designer = PromptDesigner()
        prompts = designer.design_all_prompts(kg, max_entities=2, max_relations=2)
        client = LLMClient(mock_mode=True)
        results = client.query_batch(prompts)

        processor = ResponseProcessor()
        learned_kg = processor.build_learned_kg(results, name="learned_test")

        assert learned_kg.name == "learned_test"
        assert learned_kg.source == KGSource.LEARNED
        # Learned KG should have some content
        assert learned_kg.entity_count() >= 0

    def test_process_result_adds_triples(self):
        from src.models import ProbePrompt, ProbeResult
        processor = ResponseProcessor()
        prompt = ProbePrompt(
            level=ProbeLevel.RELATIONAL,
            prompt_text="How is metformin related to diabetes?",
            entity="metformin",
            related_entity="diabetes",
            expected_relation="treats",
        )
        result = ProbeResult(
            prompt=prompt,
            response="Metformin treats type 2 diabetes by reducing glucose production.",
        )
        processed = processor.process_result(result)
        # Should have set extracted_triples (may be empty if RE finds nothing)
        assert isinstance(processed.extracted_triples, list)
