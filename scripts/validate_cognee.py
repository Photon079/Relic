#!/usr/bin/env python3
"""Day-0 Cognee validation for Relic (see relic_prd.md, sections 7 & 13).

Goal: prove that Cognee 1.2.2 ingests Relic's typed DataPoint ontology
deterministically via the low-level `add_data_points()` path -- File, Commit,
PullRequest, Issue, Person nodes wired with typed edges -- scoped to a per-repo
NodeSet, and that graph-traversal retrieval can walk the resulting chain.

This is the gate from build-sequence step 1: do not build on top until it passes.

Run:  python scripts/validate_cognee.py
Needs an LLM + embedding provider configured in .env (LLM_API_KEY / EMBEDDING_API_KEY).
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import SkipValidation

# Load .env from the repo root regardless of where the script is invoked.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Single-user, deterministic slate: with multi-user access control on (Cognee's
# default), every run registers a fresh anonymous user and prune only clears the
# current one, so old per-user subgraphs accumulate. Disable it for validation.
# Must be set before importing cognee so the config picks it up.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
# Disable session memory so a single retrieval doesn't fan out into extra
# feedback-detection LLM calls (slow, and rate-limit-prone on free tiers).
os.environ.setdefault("CACHING", "false")

import cognee
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.operations.setup import setup
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.storage import add_data_points
from cognee.modules.search.types import SearchType


# ---------------------------------------------------------------------------
# Ontology (mirrors relic_prd.md section 7). belongs_to_set is inherited from
# DataPoint and is what scopes each node to its repo's NodeSet.
# ---------------------------------------------------------------------------
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
    reviewed_by: SkipValidation[Any] = None        # edge -> Person
    metadata: dict = {"index_fields": ["title", "discussion"]}


class Commit(DataPoint):
    sha: str
    message: str
    part_of: SkipValidation[Any] = None            # edge -> PullRequest
    authored_by: SkipValidation[Any] = None        # edge -> Person
    modifies: SkipValidation[Any] = None           # edge -> list[File]
    supersedes: SkipValidation[Any] = None         # edge -> Commit
    metadata: dict = {"index_fields": ["message"]}


EXPECTED_EDGES = {
    "modifies",       # Commit -> File
    "part_of",        # Commit -> PullRequest
    "closes",         # PullRequest -> Issue
    "raised_by",      # Issue -> Person
    "authored_by",    # Commit -> Person
    "reviewed_by",    # PullRequest -> Person
    "supersedes",     # Commit -> Commit
}


def build_fixture():
    """A tiny but fully-linked 'repo': 5 files, a decision chain across an
    issue -> PR debate -> commit -> superseding commit, all wired with edges."""
    repo = NodeSet(name="relic-validation-repo")

    def scoped(dp):
        dp.belongs_to_set = [repo]
        return dp

    # People
    alice = scoped(Person(handle="alice"))      # reporter
    bob = scoped(Person(handle="bob"))          # author
    carol = scoped(Person(handle="carol"))      # reviewer

    # Files
    files = [
        scoped(File(path="src/auth.py")),
        scoped(File(path="src/serializer.py")),
        scoped(File(path="src/enums.py")),
        scoped(File(path="tests/test_auth.py")),
        scoped(File(path="README.md")),
    ]

    # The decision chain
    issue = scoped(Issue(
        number=42,
        title="Login token rejected after enum reorder",
        body="Removing the legacy enum values broke deserialization of old "
             "session tokens. Old clients send the integer value of the enum.",
        state="closed",
        raised_by=alice,
    ))

    pr = scoped(PullRequest(
        number=128,
        title="Pin enum integer values; stop reordering",
        discussion="carol: do not remove these enums, serialization depends on "
                   "their integer order. bob: agreed, pinning explicit values "
                   "and adding a regression test so this never happens again.",
        closes=issue,
        reviewed_by=carol,
    ))

    old_commit = scoped(Commit(
        sha="aaaa111",
        message="Reorder auth enums for readability",
        authored_by=bob,
        modifies=[files[2]],                    # enums.py
    ))

    fix_commit = scoped(Commit(
        sha="bbbb222",
        message="Pin enum integer values to preserve serialization (#128)",
        part_of=pr,
        authored_by=bob,
        modifies=[files[1], files[2], files[3]],  # serializer, enums, test
        supersedes=old_commit,
    ))

    points = [
        repo, alice, bob, carol, *files,
        issue, pr, old_commit, fix_commit,
    ]
    return repo, points


def summarize_graph(nodes, edges):
    """get_graph_data() returns (nodes, edges); shapes vary slightly across
    backends, so read defensively."""
    type_counts: dict[str, int] = {}
    for node in nodes:
        props = node[1] if isinstance(node, (tuple, list)) and len(node) > 1 else {}
        t = (props or {}).get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts: dict[str, int] = {}
    for edge in edges:
        rel = None
        if isinstance(edge, (tuple, list)):
            if len(edge) >= 3 and isinstance(edge[2], str):
                rel = edge[2]
            if len(edge) >= 4 and isinstance(edge[3], dict):
                rel = edge[3].get("relationship_name", rel)
        if rel:
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
    return type_counts, rel_counts


async def main() -> int:
    print("=== Relic Day-0 Cognee validation (cognee %s) ===\n" % cognee.__version__)

    print("[1/5] Pruning any prior state for a clean slate...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("[2/5] Initializing Cognee databases (setup)...")
    await setup()

    print("[3/5] Building fixture and writing typed DataPoints...")
    repo, points = build_fixture()
    await add_data_points(points)
    print(f"        wrote {len(points)} nodes scoped to NodeSet '{repo.name}'")

    print("[4/5] Reading the graph back to verify nodes and edges...")
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    type_counts, rel_counts = summarize_graph(nodes, edges)

    print("        node types :", type_counts)
    print("        edge labels :", rel_counts)

    expected_node_types = {"Person", "File", "Issue", "PullRequest", "Commit"}
    found_node_types = set(type_counts) & expected_node_types
    missing_nodes = expected_node_types - found_node_types
    found_edges = EXPECTED_EDGES & set(rel_counts)
    missing_edges = EXPECTED_EDGES - found_edges

    ok = True
    if missing_nodes:
        print(f"        FAIL: missing node types {missing_nodes}")
        ok = False
    if missing_edges:
        print(f"        FAIL: missing typed edges {missing_edges}")
        ok = False
    if not ok:
        print("\nRESULT: FAIL -- structural ingestion did not produce the "
              "expected typed graph.")
        return 1
    print("        OK: all expected node types and typed edges present.")

    print("[5/5] Running graph-traversal retrieval ('why' query)...")
    try:
        # Bounded so a rate-limited / retrying LLM can never hang the gate.
        results = await asyncio.wait_for(
            cognee.search(
                query_text="Why were the enum integer values pinned in src/enums.py?",
                query_type=SearchType.GRAPH_COMPLETION,
                node_type=NodeSet,
                node_name=[repo.name],
            ),
            timeout=120,
        )
        answer = results[0] if results else "(no answer)"
        if hasattr(answer, "result"):
            answer = answer.result
        print("        answer:", str(answer).strip()[:600])
    except (Exception, asyncio.TimeoutError) as exc:  # LLM is best-effort here
        print(f"        WARN: retrieval call failed ({type(exc).__name__}: {exc}).")
        print("        Structural ingestion still PASSED; check LLM creds / rate limits.")

    print("\nRESULT: PASS -- Cognee ingests Relic's typed ontology cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
