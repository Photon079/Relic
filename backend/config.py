"""Central config + environment setup for Relic's backend.

CRITICAL: this module sets a few `os.environ` keys that Cognee reads *at import
time*, so it must be imported before `cognee` anywhere in the process. Every
backend module that touches Cognee imports this first (`from backend import
config`). These mirror the settings validated in scripts/validate_cognee.py:

  - ENABLE_BACKEND_ACCESS_CONTROL=false -> deterministic single-user runs, so
    prune actually clears the graph instead of leaving per-user subgraphs.
  - CACHING=false -> disable session memory so a single search doesn't fan out
    into extra feedback-detection LLM calls.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env (LLM/embedding provider creds live here) before anything reads them.
load_dotenv(REPO_ROOT / ".env")

# Must be set before cognee is imported for the first time.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("CACHING", "false")

# GitHub GraphQL ingestion. A personal access token is required: the GraphQL
# API rejects unauthenticated requests outright (and a token lifts rate limits).
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

# Conservative caps for the hackathon / tiny-repo runs (no pagination yet).
DEFAULT_COMMIT_LIMIT = int(os.environ.get("RELIC_COMMIT_LIMIT", "50"))
DEFAULT_PR_LIMIT = int(os.environ.get("RELIC_PR_LIMIT", "40"))
DEFAULT_ISSUE_LIMIT = int(os.environ.get("RELIC_ISSUE_LIMIT", "40"))
