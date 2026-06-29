"""Turn graph nodes into cited, GitHub-linked provenance sources (PRD F4).

Every claim in an answer must trace back to a real PR, issue, commit, or person.
These helpers convert a node's stored properties into a human label and a real
GitHub URL, so the UI/CLI can show "no claim without a traceable source".
"""

from __future__ import annotations

import re
from typing import Any, Optional

# The typed edges that carry meaning (everything except NodeSet bookkeeping).
MEANINGFUL_EDGES = {
    "modifies",
    "part_of",
    "closes",
    "raised_by",
    "authored_by",
    "reviewed_by",
    "supersedes",
}

# GitHub handles are [A-Za-z0-9-]; anything else is a git display-name fallback
# (e.g. an author with no GitHub account) and must NOT be linked as a profile.
_HANDLE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")


def _first_line(text: Optional[str], limit: int = 80) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line[:limit] + ("…" if len(line) > limit else "")


def describe_node(props: dict[str, Any], repo_slug: str) -> dict[str, Any]:
    """Node properties -> {type, label, url} for citations and the subgraph viz."""
    ntype = props.get("type", "?")
    base = f"https://github.com/{repo_slug}"

    if ntype == "Commit":
        sha = props.get("sha") or ""
        return {
            "type": "Commit",
            "key": sha,
            "label": f"{sha[:7]} {_first_line(props.get('message'))}".strip(),
            "url": f"{base}/commit/{sha}" if sha else None,
        }
    if ntype == "PullRequest":
        num = props.get("number")
        return {
            "type": "PullRequest",
            "key": num,
            "label": f"PR #{num}: {_first_line(props.get('title'))}".strip(),
            "url": f"{base}/pull/{num}" if num is not None else None,
        }
    if ntype == "Issue":
        num = props.get("number")
        return {
            "type": "Issue",
            "key": num,
            "label": f"Issue #{num}: {_first_line(props.get('title'))}".strip(),
            "url": f"{base}/issues/{num}" if num is not None else None,
        }
    if ntype == "Person":
        handle = props.get("handle") or ""
        linkable = bool(_HANDLE_RE.match(handle))
        return {
            "type": "Person",
            "key": handle,
            "label": f"@{handle}" if linkable else handle,
            "url": f"https://github.com/{handle}" if linkable else None,
        }
    if ntype == "File":
        path = props.get("path") or ""
        return {
            "type": "File",
            "key": path,
            "label": path,
            "url": f"{base}/blob/HEAD/{path}" if path else None,
        }
    return {"type": ntype, "key": props.get("name"), "label": props.get("name") or ntype, "url": None}


# Order citations the way the decision chain reads: issue -> PR -> commit -> file -> person.
_CITATION_ORDER = {"Issue": 0, "PullRequest": 1, "Commit": 2, "File": 3, "Person": 4}


def build_citations(node_props: list[dict[str, Any]], repo_slug: str) -> list[dict[str, Any]]:
    """Dedup nodes into an ordered, GitHub-linked citation list."""
    seen: set = set()
    cites: list[dict[str, Any]] = []
    for props in node_props:
        c = describe_node(props, repo_slug)
        ident = (c["type"], c["key"])
        if ident in seen:
            continue
        seen.add(ident)
        cites.append(c)
    cites.sort(key=lambda c: (_CITATION_ORDER.get(c["type"], 9), str(c["key"])))
    return cites
