"""Evaluate routes: KG alignment, metrics, and KG graph data."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.alignment import KGAligner, MetricsCalculator
from app.session_store import get_session, update_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


@router.post("/{session_id}")
async def run_evaluation(session_id: str, threshold: float = 0.75) -> dict:
    """Align source and learned KGs and compute evaluation metrics."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.source_kg or not session.learned_kg:
        raise HTTPException(
            status_code=400,
            detail="Both source KG and learned KG are required. Run probing first.",
        )

    calc = MetricsCalculator(similarity_threshold=threshold)
    metrics = calc.evaluate(session.source_kg, session.learned_kg)

    session.metrics = metrics
    session.status = "evaluated"
    update_session(session)

    return {
        "session_id": session_id,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "correct_count": metrics.correct_count,
        "total_source": metrics.total_source,
        "total_learned": metrics.total_learned,
        "missing_count": len(metrics.missing_triples),
        "wrong_count": len(metrics.wrong_triples),
        "status": session.status,
    }


@router.get("/{session_id}/metrics")
async def get_metrics(session_id: str) -> dict:
    """Return evaluation metrics for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.metrics:
        raise HTTPException(status_code=404, detail="No metrics computed yet.")

    return session.metrics.model_dump()


@router.get("/{session_id}/source-graph")
async def get_source_graph(session_id: str) -> dict:
    """Return source KG as a D3-compatible node-link JSON."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.source_kg:
        raise HTTPException(status_code=404, detail="No source KG found.")
    return _kg_to_d3(session.source_kg, session.metrics, kg_type="source")


@router.get("/{session_id}/learned-graph")
async def get_learned_graph(session_id: str) -> dict:
    """Return learned KG as a D3-compatible node-link JSON."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.learned_kg:
        raise HTTPException(status_code=404, detail="No learned KG found.")
    return _kg_to_d3(session.learned_kg, session.metrics, kg_type="learned")


@router.get("/{session_id}/aligned-graph")
async def get_aligned_graph(session_id: str) -> dict:
    """Return combined aligned graph with match status annotations."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.source_kg or not session.learned_kg:
        raise HTTPException(status_code=404, detail="Both KGs required.")

    metrics = session.metrics

    # Build sets of matched / missing triples
    matched_source: set[tuple] = set()
    matched_learned: set[tuple] = set()
    if metrics:
        for a in metrics.alignment_details:
            if a.matched:
                matched_source.add(a.source_triple)
                if a.matched_triple:
                    matched_learned.add(a.matched_triple)

    nodes: dict[str, dict] = {}
    links: list[dict] = []

    def add_node(text: str, kg_type: str, entity_type: str = "UNKNOWN") -> None:
        key = text.lower()
        if key not in nodes:
            nodes[key] = {"id": key, "label": text, "type": entity_type, "kg": kg_type}
        elif nodes[key]["kg"] != kg_type:
            nodes[key]["kg"] = "both"

    for rel in session.source_kg.relations:
        triple = rel.triple
        status = "matched" if triple in matched_source else "missing"
        add_node(rel.head_text, "source")
        add_node(rel.tail_text, "source")
        links.append({
            "source": rel.head_text.lower(),
            "target": rel.tail_text.lower(),
            "relation": rel.relation_type,
            "kg": "source",
            "status": status,
        })

    for rel in session.learned_kg.relations:
        triple = rel.triple
        status = "matched" if triple in matched_learned else "wrong"
        add_node(rel.head_text, "learned")
        add_node(rel.tail_text, "learned")
        links.append({
            "source": rel.head_text.lower(),
            "target": rel.tail_text.lower(),
            "relation": rel.relation_type,
            "kg": "learned",
            "status": status,
        })

    return {"nodes": list(nodes.values()), "links": links}


@router.get("/{session_id}/missing-triples")
async def get_missing_triples(session_id: str) -> dict:
    """Return missing triples (knowledge gaps)."""
    session = get_session(session_id)
    if not session or not session.metrics:
        raise HTTPException(status_code=404, detail="Session or metrics not found.")
    return {
        "missing": [
            {"head": t[0], "relation": t[1], "tail": t[2]}
            for t in session.metrics.missing_triples
        ]
    }


@router.get("/{session_id}/wrong-triples")
async def get_wrong_triples(session_id: str) -> dict:
    """Return wrong triples (incorrect knowledge)."""
    session = get_session(session_id)
    if not session or not session.metrics:
        raise HTTPException(status_code=404, detail="Session or metrics not found.")
    return {
        "wrong": [
            {"head": t[0], "relation": t[1], "tail": t[2]}
            for t in session.metrics.wrong_triples
        ]
    }


def _kg_to_d3(kg, metrics, kg_type: str = "source") -> dict:
    """Convert a KnowledgeGraph to D3 node-link format."""
    matched_triples: set[tuple] = set()
    if metrics:
        if kg_type == "source":
            matched_triples = {a.source_triple for a in metrics.alignment_details if a.matched}
        else:
            matched_triples = {
                a.matched_triple
                for a in metrics.alignment_details
                if a.matched and a.matched_triple
            }

    nodes = [
        {
            "id": e.normalized,
            "label": e.text,
            "type": e.entity_type,
            "confidence": e.confidence,
        }
        for e in kg.entities.values()
    ]

    links = []
    for rel in kg.relations:
        triple = rel.triple
        status = "matched" if triple in matched_triples else (
            "missing" if kg_type == "source" else "wrong"
        )
        links.append({
            "source": rel.head_text.lower(),
            "target": rel.tail_text.lower(),
            "relation": rel.relation_type,
            "confidence": rel.confidence,
            "status": status,
        })

    return {"nodes": nodes, "links": links}
