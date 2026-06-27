"""Relic's typed knowledge-graph ontology (PRD section 7).

These are the *exact* DataPoint models proven in scripts/validate_cognee.py,
now the single source of truth for the real ingestion pipeline. Assigning one
DataPoint to another's field creates a typed edge whose label is the field name;
`index_fields` marks the text that gets embedded for semantic search.

IDs are deterministic (uuid5 over repo + type + natural key) so re-ingesting a
repo updates the same nodes instead of creating duplicates, and so dedup of
shared nodes (a Person, a File) across many edges is just dict lookup.
"""

import uuid
from typing import Any

from pydantic import SkipValidation

# Importing backend.config first guarantees Cognee's env knobs are set before
# the cognee import below runs (see backend/config.py).
from backend import config  # noqa: F401

from cognee.infrastructure.engine import DataPoint

# Stable namespace for deterministic node IDs.
_RELIC_NS = uuid.uuid5(uuid.NAMESPACE_URL, "https://relic.dev/graph")


def make_id(repo: str, node_type: str, key: str) -> uuid.UUID:
    """Deterministic node id: same (repo, type, natural key) -> same UUID."""
    return uuid.uuid5(_RELIC_NS, f"{repo}|{node_type}|{key}")


class Person(DataPoint):
    handle: str
    metadata: dict = {"index_fields": ["handle"]}


class File(DataPoint):
    path: str
    metadata: dict = {"index_fields": ["path"]}


class Issue(DataPoint):
    number: int
    title: str
    body: str
    state: str = "closed"
    raised_by: SkipValidation[Any] = None          # edge -> Person
    metadata: dict = {"index_fields": ["title", "body"]}


class PullRequest(DataPoint):
    number: int
    title: str
    discussion: str
    closes: SkipValidation[Any] = None             # edge -> Issue
    reviewed_by: SkipValidation[Any] = None        # edge -> list[Person]
    metadata: dict = {"index_fields": ["title", "discussion"]}


class Commit(DataPoint):
    sha: str
    message: str
    part_of: SkipValidation[Any] = None            # edge -> PullRequest
    authored_by: SkipValidation[Any] = None        # edge -> Person
    modifies: SkipValidation[Any] = None           # edge -> list[File]
    supersedes: SkipValidation[Any] = None         # edge -> Commit
    metadata: dict = {"index_fields": ["message"]}


# The typed edges the graph should contain once a repo is ingested. Used by the
# builder's read-back verification.
EXPECTED_EDGES = {
    "modifies",       # Commit -> File
    "part_of",        # Commit -> PullRequest
    "closes",         # PullRequest -> Issue
    "raised_by",      # Issue -> Person
    "authored_by",    # Commit -> Person
    "reviewed_by",    # PullRequest -> Person
    "supersedes",     # Commit -> Commit
}
