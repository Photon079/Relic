"""Build a repo's typed knowledge graph in Cognee (PRD F2).

Same low-level flow proven in scripts/validate_cognee.py -- create a per-repo
NodeSet, scope every node to it, and write the structure deterministically with
`add_data_points()` -- but fed by real GitHub data via the ingestion layer
instead of a synthetic fixture.

Run end to end:
    python -m backend.graph.builder <owner/name or github url> [--prune]
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from backend import config  # noqa: F401  (sets Cognee env before cognee import)
from backend.graph.models import EXPECTED_EDGES
from backend.ingestion.github_client import fetch_repo, parse_repo_url
from backend.ingestion.mappers import map_repository

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.operations.setup import setup
from cognee.tasks.storage import add_data_points


async def build_graph(repo_url: str, *, prune: bool = False) -> dict[str, Any]:
    """Fetch, map, and write one repo into its own NodeSet. Returns a summary."""
    owner, name = parse_repo_url(repo_url)
    slug = f"{owner}/{name}"
    print(f"[1/5] Fetching {slug} via GitHub GraphQL...")
    repository = await fetch_repo(owner, name)

    print("[2/5] Mapping GraphQL response into typed DataPoints...")
    points = map_repository(repository)
    if not points:
        raise RuntimeError(f"No graph data produced for {slug}.")

    # Scope every node to this repo's NodeSet so graphs never bleed across repos.
    node_set = NodeSet(name=f"repo:{slug}")
    for point in points:
        point.belongs_to_set = [node_set]
    print(f"        mapped {len(points)} nodes -> NodeSet '{node_set.name}'")

    if prune:
        print("        pruning prior Cognee state...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    print("[3/5] Initializing Cognee databases (setup)...")
    await setup()

    print("[4/5] Writing DataPoints with add_data_points()...")
    await add_data_points([node_set, *points])

    print("[5/5] Reading the graph back to verify nodes and edges...")
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    type_counts, rel_counts = _summarize(nodes, edges)
    print("        node types :", type_counts)
    print("        edge labels :", rel_counts)

    present_edges = sorted(EXPECTED_EDGES & set(rel_counts))
    missing_edges = sorted(EXPECTED_EDGES - set(rel_counts))
    print(f"        typed edges present : {present_edges}")
    if missing_edges:
        # Not necessarily a failure: a repo may simply lack, e.g., reviews.
        print(f"        typed edges absent  : {missing_edges} "
              "(repo may not contain that relationship)")

    return {
        "repo": slug,
        "node_set": node_set.name,
        "node_count": len(points),
        "node_types": type_counts,
        "edge_labels": rel_counts,
        "edges_present": present_edges,
        "edges_absent": missing_edges,
    }


def _summarize(nodes, edges) -> tuple[dict[str, int], dict[str, int]]:
    """Count nodes by `type` and edges by relationship label (defensively)."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a repo's Relic graph in Cognee.")
    parser.add_argument("repo", help="GitHub repo as 'owner/name' or a full URL")
    parser.add_argument(
        "--prune", action="store_true",
        help="Wipe all prior Cognee data first (clean slate for the whole DB)",
    )
    args = parser.parse_args()

    summary = asyncio.run(build_graph(args.repo, prune=args.prune))
    print(f"\nRESULT: built graph for {summary['repo']} "
          f"({summary['node_count']} nodes, "
          f"{len(summary['edges_present'])}/{len(EXPECTED_EDGES)} edge types present).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
