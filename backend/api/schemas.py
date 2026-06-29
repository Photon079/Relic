"""Request/response models for the Relic HTTP API.

These only describe the wire shape. The response bodies mirror exactly what
backend.graph.builder and backend.retrieval.query already return — the routes
do not reshape that data.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# --- /ingest ----------------------------------------------------------------
class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repo as 'owner/name' or a full URL")


class IngestResponse(BaseModel):
    repo: str
    node_set: str
    node_count: int
    node_types: dict[str, int]
    edge_labels: dict[str, int]


# --- /query -----------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="Plain-English 'why' question")
    repo_url: Optional[str] = Field(
        None, description="Optional 'owner/name' or URL to scope the query to one repo"
    )


class QueryResponse(BaseModel):
    # Exact shape produced by backend.retrieval.query.answer_question.
    answer: str
    chain: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    subgraph: dict[str, Any]


# --- /health ----------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
