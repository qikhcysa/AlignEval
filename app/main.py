"""AlignEval FastAPI application entry point."""
from __future__ import annotations

import json
import os
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes.sessions import router as sessions_router
from app.routes.probe import router as probe_router
from app.routes.evaluate import router as evaluate_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlignEval",
    description="Knowledge Alignment Evaluation for LLM Knowledge Defect Detection",
    version="1.0.0",
)

# Static files & templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Register routers
app.include_router(sessions_router)
app.include_router(probe_router)
app.include_router(evaluate_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/graph/{session_id}", response_class=HTMLResponse)
async def graph_view(request: Request, session_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "graph.html", {"session_id": session_id}
    )


@app.get("/evaluation/{session_id}", response_class=HTMLResponse)
async def evaluation_view(request: Request, session_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "evaluation.html", {"session_id": session_id}
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/sample-dataset")
async def sample_dataset() -> dict:
    """Return the bundled sample dataset for quick demo."""
    data_path = BASE_DIR.parent / "data" / "sample_qa.json"
    if data_path.exists():
        with open(data_path) as f:
            return json.load(f)
    return {"data": []}
