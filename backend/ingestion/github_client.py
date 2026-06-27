"""GitHub GraphQL ingestion (PRD F1).

One GraphQL query pulls the linked history Relic needs -- commits, pull
requests, review comments, issues, and the references that tie them together
(commit->PR via associatedPullRequests, PR->issue via closingIssuesReferences) --
in a single round trip. GraphQL is the point: it returns the relationships flat
REST would make us stitch together client-side.

GraphQL note: the Commit object has no per-commit `files` field, so the files a
commit touched are taken from its pull request's `files` connection (see
mappers.py). That keeps ingestion to one GraphQL call with no REST fan-out.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from backend import config

# Single query. Connection sizes are passed as variables so caller can cap them.
_REPO_QUERY = """
query Repo($owner: String!, $name: String!, $commits: Int!, $prs: Int!, $issues: Int!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    defaultBranchRef {
      name
      target {
        ... on Commit {
          history(first: $commits) {
            nodes {
              oid
              message
              committedDate
              author { name user { login } }
              associatedPullRequests(first: 5) { nodes { number } }
            }
          }
        }
      }
    }
    pullRequests(first: $prs, orderBy: { field: UPDATED_AT, direction: DESC }) {
      nodes {
        number
        title
        body
        state
        author { login }
        reviews(first: 50) { nodes { author { login } body state } }
        comments(first: 50) { nodes { author { login } body } }
        files(first: 100) { nodes { path } }
        closingIssuesReferences(first: 10) { nodes { number } }
        commits(first: 100) { nodes { commit { oid } } }
      }
    }
    issues(first: $issues, orderBy: { field: UPDATED_AT, direction: DESC }) {
      nodes {
        number
        title
        body
        state
        author { login }
      }
    }
  }
}
"""

_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com[/:]([^/]+)/([^/#?]+?)(?:\.git)?/?$"
)


class GitHubError(RuntimeError):
    """Raised when the GitHub GraphQL API returns an error or no token is set."""


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """'https://github.com/owner/name' (or 'owner/name') -> ('owner', 'name')."""
    repo_url = repo_url.strip()
    match = _URL_RE.match(repo_url)
    if match:
        return match.group(1), match.group(2)
    if "/" in repo_url and " " not in repo_url:  # bare "owner/name"
        owner, _, name = repo_url.partition("/")
        if owner and name:
            return owner, name.rstrip("/")
    raise GitHubError(f"Could not parse a GitHub owner/name from: {repo_url!r}")


async def fetch_repo(
    owner: str,
    name: str,
    *,
    commit_limit: int = config.DEFAULT_COMMIT_LIMIT,
    pr_limit: int = config.DEFAULT_PR_LIMIT,
    issue_limit: int = config.DEFAULT_ISSUE_LIMIT,
    token: str | None = None,
) -> dict[str, Any]:
    """Run the GraphQL query and return the raw `repository` object."""
    token = token or config.GITHUB_TOKEN
    if not token:
        raise GitHubError(
            "No GitHub token. Set GITHUB_TOKEN in .env -- the GraphQL API "
            "rejects unauthenticated requests."
        )

    variables = {
        "owner": owner,
        "name": name,
        "commits": commit_limit,
        "prs": pr_limit,
        "issues": issue_limit,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "relic-ingestion",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            config.GITHUB_GRAPHQL_URL,
            json={"query": _REPO_QUERY, "variables": variables},
            headers=headers,
        )

    if resp.status_code == 401:
        raise GitHubError("GitHub rejected the token (401). Check GITHUB_TOKEN.")
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("errors"):
        raise GitHubError(f"GraphQL errors: {payload['errors']}")
    repository = (payload.get("data") or {}).get("repository")
    if repository is None:
        raise GitHubError(
            f"Repository {owner}/{name} not found or not accessible with this token."
        )
    return repository
