"""In-memory session store for the web app."""
from __future__ import annotations

from src.models import EvaluationSession

_sessions: dict[str, EvaluationSession] = {}


def get_session(session_id: str) -> EvaluationSession | None:
    return _sessions.get(session_id)


def create_session(session: EvaluationSession) -> EvaluationSession:
    _sessions[session.id] = session
    return session


def update_session(session: EvaluationSession) -> EvaluationSession:
    _sessions[session.id] = session
    return session


def list_sessions() -> list[EvaluationSession]:
    return list(_sessions.values())


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False
