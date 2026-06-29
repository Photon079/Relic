"""The "why" query: walk the decision chain and compose a grounded answer (PRD F3).

Pipeline:
  1. Cognee vector retrieval finds the entry nodes most relevant to the question.
  2. We walk the graph outward from those seeds along the *meaningful* typed edges
     (modifies / part_of / closes / raised_by / authored_by / reviewed_by /
     supersedes), skipping NodeSet bookkeeping, to assemble the provenance chain.
  3. The Groq LLM composes an answer grounded ONLY in that walked subgraph, so it
     can't hallucinate beyond what the graph actually says.
  4. Every node in the chain becomes a cited, GitHub-linked source.

This is the multi-hop graph walk flat vector RAG structurally cannot do.

Run:  python -m backend.retrieval.query "why ...?" [--repo owner/name] [--top-k N]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from typing import Any, Optional

from backend import config  # noqa: F401  (sets Cognee env before cognee import)
from backend.retrieval.citations import MEANINGFUL_EDGES, build_citations, describe_node

import litellm
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.config import get_llm_config
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever

_MAX_HOPS = 4
_MAX_SUBGRAPH_NODES = 30
_MAX_SEEDS = 8

# Only these node types are expanded during the walk. File and Person are hubs
# (every commit modifies the same few files), so we include them as terminal
# leaves but never traverse *through* them — otherwise the walk floods the repo.
_EXPANDABLE = {"Commit", "PullRequest", "Issue"}

_SYSTEM_PROMPT = (
    "You are Relic. You explain WHY code is the way it is, using only a provenance "
    "subgraph extracted from a repository's history (commits, pull requests, issues, "
    "and people). Answer the question strictly from the provided facts. Do not invent "
    "PRs, issues, commits, or names. If the facts are insufficient, say so plainly. "
    "Reference specifics (PR numbers, issue numbers, short commit SHAs, handles)."
)


# --------------------------------------------------------------------------- #
# Graph loading / normalization
# --------------------------------------------------------------------------- #
async def _load_graph() -> tuple[dict[str, dict], list[tuple[str, str, str]], dict[str, str]]:
    """Return (props_by_id, normalized_edges, nodeset_name_to_id)."""
    engine = await get_graph_engine()
    nodes, edges = await engine.get_graph_data()

    props_by_id = {str(nid): (props or {}) for nid, props in nodes}
    nodeset_ids = {
        props.get("name"): str(nid)
        for nid, props in nodes
        if (props or {}).get("type") == "NodeSet" and (props or {}).get("name")
    }
    norm = [e for e in (_normalize_edge(edge) for edge in edges) if e]
    return props_by_id, norm, nodeset_ids


def _normalize_edge(edge: Any) -> Optional[tuple[str, str, str]]:
    if not isinstance(edge, (tuple, list)) or len(edge) < 3:
        return None
    src, tgt = str(edge[0]), str(edge[1])
    rel = edge[2] if isinstance(edge[2], str) else None
    if len(edge) >= 4 and isinstance(edge[3], dict):
        rel = edge[3].get("relationship_name", rel)
    return (src, tgt, rel) if rel else None


def _node_id(node: Any) -> Optional[str]:
    nid = getattr(node, "id", None)
    if nid is None and hasattr(node, "attributes"):
        nid = (node.attributes or {}).get("id")
    return str(nid) if nid is not None else None


# --------------------------------------------------------------------------- #
# Retrieval + traversal
# --------------------------------------------------------------------------- #
async def _entry_node_ids(question: str, top_k: int) -> list[str]:
    """Vector search over the graph -> ids of the most relevant nodes, ranked."""
    retriever = GraphCompletionRetriever(top_k=top_k)
    triplets = await retriever.get_triplets(question)
    ordered: list[str] = []
    seen: set[str] = set()
    for edge in triplets or []:
        for node in (getattr(edge, "node1", None), getattr(edge, "node2", None)):
            nid = _node_id(node)
            if nid and nid not in seen:
                seen.add(nid)
                ordered.append(nid)
    return ordered


def _walk(
    seeds: list[str],
    edges: list[tuple[str, str, str]],
    props_by_id: dict[str, dict],
    members: Optional[set[str]],
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """BFS outward from seeds along meaningful edges -> (subgraph nodes, chain edges).

    File/Person nodes are included but never expanded through (see _EXPANDABLE),
    so the walk follows the decision chain (commit -> PR -> issue) instead of
    flooding the whole repo via shared-file hubs.
    """
    meaningful = [
        (s, t, r)
        for s, t, r in edges
        if r in MEANINGFUL_EDGES
        and s in props_by_id
        and t in props_by_id
        and (members is None or (s in members and t in members))
    ]
    adj: dict[str, list[str]] = defaultdict(list)
    for s, t, _ in meaningful:
        adj[s].append(t)
        adj[t].append(s)

    def in_scope(n: str) -> bool:
        return n in props_by_id and (members is None or n in members)

    seed_list = [s for s in seeds if in_scope(s)][:_MAX_SEEDS]
    visited = set(seed_list)
    frontier = list(seed_list)
    for _ in range(_MAX_HOPS):
        nxt: list[str] = []
        for node in frontier:
            if props_by_id.get(node, {}).get("type") not in _EXPANDABLE:
                continue  # terminal leaf (File/Person): don't traverse through it
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
                    if len(visited) >= _MAX_SUBGRAPH_NODES:
                        break
            if len(visited) >= _MAX_SUBGRAPH_NODES:
                break
        frontier = nxt
        if not frontier or len(visited) >= _MAX_SUBGRAPH_NODES:
            break

    chain = [(s, t, r) for s, t, r in meaningful if s in visited and t in visited]
    return visited, chain


def _build_chain_and_subgraph(
    chain_edges: list[tuple[str, str, str]],
    subgraph_nodes: set[str],
    props_by_id: dict[str, dict],
    slug: str,
) -> tuple[list[dict], dict, str]:
    """Render the walked subgraph as structured chain, viz subgraph, and LLM text."""
    chain: list[dict] = []
    text_lines: list[str] = []
    for s, t, rel in chain_edges:
        src = describe_node(props_by_id[s], slug)
        tgt = describe_node(props_by_id[t], slug)
        chain.append({"source": src, "relation": rel, "target": tgt})
        text_lines.append(f"{src['label']} --{rel}--> {tgt['label']}")

    subgraph = {
        "nodes": [
            {"id": nid, **describe_node(props_by_id[nid], slug)} for nid in subgraph_nodes
        ],
        "edges": [{"source": s, "target": t, "relation": r} for s, t, r in chain_edges],
    }
    return chain, subgraph, "\n".join(text_lines)


# --------------------------------------------------------------------------- #
# Answer composition (Groq, grounded only in the chain)
# --------------------------------------------------------------------------- #
async def _compose_answer(question: str, chain_text: str) -> str:
    if not chain_text.strip():
        return ("I couldn't find a provenance chain in the graph for that question — "
                "no linked commits, PRs, or issues matched.")
    cfg = get_llm_config()
    user = (
        f"Question: {question}\n\n"
        f"Provenance subgraph (the only facts you may use):\n{chain_text}\n\n"
        "Explain why, citing the specific PRs / issues / commits / people above."
    )
    resp = await litellm.acompletion(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def _resolve_repo(repo: Optional[str], nodeset_ids: dict[str, str]) -> tuple[str, Optional[str]]:
    """Pick the target repo NodeSet -> (slug, nodeset_id). repo may be 'owner/name'."""
    if repo:
        want = f"repo:{repo}"
        if want in nodeset_ids:
            return repo, nodeset_ids[want]
        return repo, None  # not found; proceed unscoped
    repo_sets = {k: v for k, v in nodeset_ids.items() if k.startswith("repo:")}
    if len(repo_sets) == 1:
        name, nid = next(iter(repo_sets.items()))
        return name[len("repo:"):], nid
    if not repo_sets:
        return "unknown/repo", None
    raise ValueError(
        f"Multiple repos in the graph; pass --repo one of: "
        f"{[k[len('repo:'):] for k in repo_sets]}"
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def answer_question(question: str, repo: Optional[str] = None, top_k: int = 10) -> dict[str, Any]:
    props_by_id, edges, nodeset_ids = await _load_graph()
    slug, nodeset_id = _resolve_repo(repo, nodeset_ids)

    members: Optional[set[str]] = None
    if nodeset_id is not None:
        members = {s for s, t, r in edges if r == "belongs_to_set" and t == nodeset_id}

    seeds = await _entry_node_ids(question, top_k)
    subgraph_nodes, chain_edges = _walk(seeds, edges, props_by_id, members)
    chain, subgraph, chain_text = _build_chain_and_subgraph(
        chain_edges, subgraph_nodes, props_by_id, slug
    )
    answer = await _compose_answer(question, chain_text)
    citations = build_citations([props_by_id[n] for n in subgraph_nodes], slug)

    return {
        "question": question,
        "repo": slug,
        "answer": answer,
        "chain": chain,
        "citations": citations,
        "subgraph": subgraph,
    }


def _print(result: dict[str, Any]) -> None:
    print(f"\nQ: {result['question']}  [repo: {result['repo']}]\n")
    print("ANSWER:\n" + result["answer"] + "\n")
    if result["chain"]:
        print(f"PROVENANCE CHAIN ({len(result['chain'])} edges):")
        for link in result["chain"]:
            print(f"  • {link['source']['label']}  --{link['relation']}-->  {link['target']['label']}")
    print(f"\nCITATIONS ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  [{c['type']}] {c['label']}" + (f"  {c['url']}" if c["url"] else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask Relic why code is the way it is.")
    parser.add_argument("question", help="Plain-English 'why' question")
    parser.add_argument("--repo", help="owner/name (needed if multiple repos are ingested)")
    parser.add_argument("--top-k", type=int, default=10, help="entry nodes from vector search")
    args = parser.parse_args()
    result = asyncio.run(answer_question(args.question, repo=args.repo, top_k=args.top_k))
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
