"""Upload & session management routes."""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.models import EvaluationSession
from src.kg_builder import KGConstructor
from app.session_store import create_session, list_sessions, get_session, delete_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionCreate(BaseModel):
    name: str
    domain: str = "general"
    model_name: str = "gpt-4o-mini"


@router.post("/")
async def new_session(body: SessionCreate) -> dict:
    """Create a new evaluation session."""
    session = EvaluationSession(
        name=body.name,
        domain=body.domain,
        model_name=body.model_name,
        status="pending",
    )
    create_session(session)
    return {"session_id": session.id, "name": session.name, "status": session.status}


@router.get("/")
async def get_sessions() -> list[dict]:
    """List all sessions."""
    sessions = list_sessions()
    return [
        {
            "session_id": s.id,
            "name": s.name,
            "status": s.status,
            "domain": s.domain,
            "model_name": s.model_name,
            "source_kg_entities": s.source_kg.entity_count() if s.source_kg else 0,
            "source_kg_relations": s.source_kg.relation_count() if s.source_kg else 0,
            "learned_kg_entities": s.learned_kg.entity_count() if s.learned_kg else 0,
            "learned_kg_relations": s.learned_kg.relation_count() if s.learned_kg else 0,
        }
        for s in sessions
    ]


@router.get("/{session_id}")
async def get_session_detail(session_id: str) -> dict:
    """Get session details."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.id,
        "name": session.name,
        "status": session.status,
        "domain": session.domain,
        "model_name": session.model_name,
        "source_kg_entities": session.source_kg.entity_count() if session.source_kg else 0,
        "source_kg_relations": session.source_kg.relation_count() if session.source_kg else 0,
        "learned_kg_entities": session.learned_kg.entity_count() if session.learned_kg else 0,
        "learned_kg_relations": session.learned_kg.relation_count() if session.learned_kg else 0,
        "probe_count": len(session.probe_results),
        "metrics": session.metrics.model_dump() if session.metrics else None,
    }


@router.delete("/{session_id}")
async def remove_session(session_id: str) -> dict:
    if delete_session(session_id):
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/{session_id}/upload-dataset")
async def upload_dataset(
    session_id: str,
    file: UploadFile = File(...),
    domain: str = Form("general"),
) -> dict:
    """Upload a Q&A dataset JSON file and build the source knowledge graph."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    content = await file.read()
    try:
        data: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "data" in data:
        records = data["data"]
    else:
        raise HTTPException(
            status_code=400,
            detail="Expected JSON array or object with 'data' key containing Q&A records.",
        )

    constructor = KGConstructor(source="source")
    source_kg = constructor.build_from_dicts(records, name=f"source_kg_{session_id[:8]}")

    session.source_kg = source_kg
    session.domain = domain
    session.status = "dataset_uploaded"

    from app.session_store import update_session
    update_session(session)

    return {
        "session_id": session_id,
        "entity_count": source_kg.entity_count(),
        "relation_count": source_kg.relation_count(),
        "status": session.status,
    }
