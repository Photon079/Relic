"""Map raw GitHub GraphQL JSON into Relic's wired DataPoint instances (PRD F2).

The whole value is in the edges, so this module's job is to turn GitHub's nested
JSON into shared, deduplicated nodes (one Person per handle, one File per path)
connected by the typed edges in graph/models.py:

    Issue   -> raised_by    -> Person
    PR      -> closes       -> Issue
    PR      -> reviewed_by  -> [Person]
    Commit  -> authored_by  -> Person
    Commit  -> part_of      -> PullRequest
    Commit  -> modifies     -> [File]      (files come from the commit's PR)
    Commit  -> supersedes   -> [Commit]    (computed: prior change to same file)
"""

from __future__ import annotations

from typing import Any, Optional

from backend.graph.models import (
    Commit,
    File,
    Issue,
    Person,
    PullRequest,
    make_id,
)


def _text(value: Optional[str], fallback: str) -> str:
    """Index fields must never be empty: Gemini's embedder 422s on "".

    Coalesce blank text to a meaningful, always-non-empty fallback (e.g. the
    node's number/sha) so every embeddable field carries real content.
    """
    value = (value or "").strip()
    return value if value else fallback


def _login(author: Optional[dict]) -> Optional[str]:
    """Pull a usable handle from a GraphQL author/user object (may be None)."""
    if not author:
        return None
    user = author.get("user")
    if user and user.get("login"):
        return user["login"]
    return author.get("login") or author.get("name")


class _Registry:
    """Creates-or-returns shared nodes so edges point at one instance per key."""

    def __init__(self, repo: str):
        self.repo = repo
        self.persons: dict[str, Person] = {}
        self.files: dict[str, File] = {}
        self.issues: dict[int, Issue] = {}
        self.prs: dict[int, PullRequest] = {}
        self.commits: dict[str, Commit] = {}

    def person(self, handle: Optional[str]) -> Optional[Person]:
        if not handle:
            return None
        if handle not in self.persons:
            self.persons[handle] = Person(
                id=make_id(self.repo, "Person", handle), handle=handle
            )
        return self.persons[handle]

    def file(self, path: str) -> File:
        if path not in self.files:
            self.files[path] = File(id=make_id(self.repo, "File", path), path=path)
        return self.files[path]

    def all_points(self) -> list[Any]:
        return [
            *self.persons.values(),
            *self.files.values(),
            *self.issues.values(),
            *self.prs.values(),
            *self.commits.values(),
        ]


def map_repository(repository: dict[str, Any]) -> list[Any]:
    """Raw `repository` GraphQL object -> flat list of wired DataPoints."""
    repo = repository.get("nameWithOwner") or "unknown/repo"
    reg = _Registry(repo)

    # --- Issues -> raised_by -> Person ------------------------------------
    for node in _nodes(repository.get("issues")):
        number = node["number"]
        title = _text(node.get("title"), f"Issue #{number}")
        reg.issues[number] = Issue(
            id=make_id(repo, "Issue", str(number)),
            number=number,
            title=title,
            body=_text(node.get("body"), title),  # empty body -> reuse title
            state=(node.get("state") or "OPEN").lower(),
            raised_by=reg.person(_login(node.get("author"))),
        )

    # --- Pull requests: discussion, closes, reviewers, file set -----------
    # pr_files: PR number -> [File]; used to fill Commit.modifies below.
    pr_files: dict[int, list[File]] = {}
    for node in _nodes(repository.get("pullRequests")):
        number = node["number"]

        discussion = _concat_discussion(node)
        reviewers = _dedup(
            reg.person(_login(r.get("author"))) for r in _nodes(node.get("reviews"))
        )
        closed_issues = _dedup(
            reg.issues.get(ref["number"])
            for ref in _nodes(node.get("closingIssuesReferences"))
        )
        files = [reg.file(f["path"]) for f in _nodes(node.get("files")) if f.get("path")]
        pr_files[number] = files

        title = _text(node.get("title"), f"PR #{number}")
        reg.prs[number] = PullRequest(
            id=make_id(repo, "PullRequest", str(number)),
            number=number,
            title=title,
            discussion=_text(discussion, title),  # silent PR -> reuse title
            closes=closed_issues or None,
            reviewed_by=reviewers or None,
        )

    # --- Commits: authored_by, part_of, modifies --------------------------
    # Ordering metadata is kept out-of-band (pydantic models reject stray attrs).
    commit_meta: dict[str, dict[str, Any]] = {}
    history = (
        ((repository.get("defaultBranchRef") or {}).get("target") or {}).get("history")
    )
    for node in _nodes(history):
        oid = node["oid"]
        pr_numbers = [
            ref["number"]
            for ref in _nodes(node.get("associatedPullRequests"))
            if ref["number"] in reg.prs
        ]
        part_of = reg.prs[pr_numbers[0]] if pr_numbers else None
        modifies = _dedup(f for n in pr_numbers for f in pr_files.get(n, []))

        reg.commits[oid] = Commit(
            id=make_id(repo, "Commit", oid),
            sha=oid,
            message=_text(node.get("message"), f"Commit {oid[:7]}"),
            authored_by=reg.person(_login(node.get("author"))),
            part_of=part_of,
            modifies=modifies or None,
        )
        commit_meta[oid] = {
            "date": node.get("committedDate") or "",
            "paths": [f.path for f in modifies],
        }

    _wire_supersedes(reg, commit_meta)
    return reg.all_points()


def _wire_supersedes(reg: _Registry, commit_meta: dict[str, dict[str, Any]]) -> None:
    """Commit -> supersedes -> the previous commit that touched the same file.

    For each file, order the commits that modified it oldest-first; each commit
    supersedes its immediate predecessor on that file. A commit touching several
    files can supersede several commits (list edge), deduped.
    """
    by_file: dict[str, list[str]] = {}
    for oid, meta in commit_meta.items():
        for path in meta["paths"]:
            by_file.setdefault(path, []).append(oid)

    predecessors: dict[str, list[Commit]] = {}
    for oids in by_file.values():
        oids.sort(key=lambda o: commit_meta[o]["date"])
        for newer, older in zip(oids[1:], oids[:-1]):
            predecessors.setdefault(newer, []).append(reg.commits[older])

    for oid, olders in predecessors.items():
        reg.commits[oid].supersedes = _dedup(olders) or None


def _concat_discussion(pr_node: dict[str, Any]) -> str:
    """Flatten review + comment bodies into one searchable discussion string."""
    parts: list[str] = []
    if pr_node.get("body"):
        parts.append(pr_node["body"])
    for review in _nodes(pr_node.get("reviews")):
        handle = _login(review.get("author")) or "someone"
        body = (review.get("body") or "").strip()
        if body:
            parts.append(f"{handle} ({(review.get('state') or '').lower()}): {body}")
    for comment in _nodes(pr_node.get("comments")):
        handle = _login(comment.get("author")) or "someone"
        body = (comment.get("body") or "").strip()
        if body:
            parts.append(f"{handle}: {body}")
    return "\n".join(parts)


def _nodes(connection: Optional[dict]) -> list[dict]:
    """Safely unwrap a GraphQL `{ nodes: [...] }` connection (may be None)."""
    if not connection:
        return []
    return [n for n in (connection.get("nodes") or []) if n]


def _dedup(items) -> list:
    """Order-preserving dedup of DataPoints by id, dropping None."""
    seen: set = set()
    out: list = []
    for item in items:
        if item is None or item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
    return out
