"""POST /query — thin HTTP wrapper over backend.retrieval.query.answer_question.

No retrieval logic here. It optionally normalizes a repo URL to 'owner/name'
(reusing the ingestion parser) and returns the retrieval output unchanged:
{answer, chain, citations, subgraph}.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend import config  # noqa: F401  (sets Cognee env before cognee import)
from backend.api.schemas import QueryRequest, QueryResponse
from backend.ingestion.github_client import GitHubError, parse_repo_url
from backend.retrieval.query import answer_question

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    repo = None
    if req.repo_url:
        try:
            owner, name = parse_repo_url(req.repo_url)
            repo = f"{owner}/{name}"
        except GitHubError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = await answer_question(req.question, repo=repo)
    except ValueError as exc:  # e.g. multiple repos ingested, none specified
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result["answer"],
        chain=result["chain"],
        citations=result["citations"],
        subgraph=result["subgraph"],
    )
