"""Probe routes: run multi-level LLM probing."""
from __future__ import annotations

import logging
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from src.probing import PromptDesigner, LLMClient, ResponseProcessor
from app.session_store import get_session, update_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/probe", tags=["probe"])


class ProbeConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    mock_mode: bool = True
    max_entities: int = 30
    max_relations: int = 30
    temperature: float = 0.2


@router.post("/{session_id}")
async def run_probe(session_id: str, config: ProbeConfig) -> dict:
    """Run multi-level probing against the source KG entities and build learned KG."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.source_kg:
        raise HTTPException(status_code=400, detail="No source KG found. Upload a dataset first.")

    # Design prompts
    designer = PromptDesigner(domain=session.domain, seed=42)
    prompts = designer.design_all_prompts(
        session.source_kg,
        max_entities=config.max_entities,
        max_relations=config.max_relations,
    )

    # Query LLM
    llm = LLMClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        mock_mode=config.mock_mode,
        temperature=config.temperature,
    )
    results = llm.query_batch(prompts)

    # Build learned KG from responses
    processor = ResponseProcessor()
    learned_kg = processor.build_learned_kg(
        results, name=f"learned_kg_{session_id[:8]}"
    )

    session.probe_results = results
    session.learned_kg = learned_kg
    session.model_name = config.model if not config.mock_mode else "mock"
    session.status = "probed"
    update_session(session)

    return {
        "session_id": session_id,
        "prompt_count": len(prompts),
        "learned_entities": learned_kg.entity_count(),
        "learned_relations": learned_kg.relation_count(),
        "status": session.status,
    }


@router.get("/{session_id}/prompts")
async def get_prompts(session_id: str) -> dict:
    """Return the probe prompts and responses for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "probe_results": [
            {
                "id": r.id,
                "level": r.prompt.level.value,
                "prompt": r.prompt.prompt_text,
                "entity": r.prompt.entity,
                "related_entity": r.prompt.related_entity,
                "response": r.response,
                "triples": r.extracted_triples,
                "latency_ms": r.latency_ms,
            }
            for r in session.probe_results
        ],
    }
