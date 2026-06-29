"""Relic FastAPI app — HTTP surface over the ingestion + retrieval pipeline.

This module only wires routes and middleware; all real work lives in
backend.graph.builder and backend.retrieval.query.

`backend.config` is imported first so Cognee's env knobs are set before any
cognee import is triggered (transitively via the routers).

Run:  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from backend import config  # noqa: F401  MUST be imported before cognee

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_ingest import router as ingest_router
from backend.api.routes_query import router as query_router
from backend.api.schemas import HealthResponse

app = FastAPI(
    title="Relic API",
    description="Walk a repo's decision chain: why is this code the way it is?",
    version="0.1.0",
)

# Permissive CORS so the React frontend (different port/origin) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # cannot combine credentials with wildcard origin
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


app.include_router(ingest_router)
app.include_router(query_router)
