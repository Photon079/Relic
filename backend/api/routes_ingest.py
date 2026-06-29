"""POST /ingest — thin HTTP wrapper over backend.graph.builder.build_graph.

No ingestion/graph logic lives here; it just awaits the existing builder and
returns a summary. Ingestion is slow, so the request blocks until it completes
(no background jobs — fine for the demo).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend import config  # noqa: F401  (sets Cognee env before cognee import)
from backend.api.schemas import IngestRequest, IngestResponse
from backend.graph.builder import build_graph

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    try:
        summary = await build_graph(req.repo_url)
    except Exception as exc:  # bad URL, missing token, GitHub/Cognee failure
        raise HTTPException(status_code=400, detail=str(exc))

    return IngestResponse(
        repo=summary["repo"],
        node_set=summary["node_set"],
        node_count=summary["node_count"],
        node_types=summary["node_types"],
        edge_labels=summary["edge_labels"],
    )
