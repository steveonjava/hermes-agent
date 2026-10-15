#!/usr/bin/env python3
"""Check only fork changes above a trusted official stable release."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

# Script execution puts scripts/, rather than the repository root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci.check_blob_size import GitError, allowlist_at, find_offenders

OFFICIAL_URL = "https://github.com/NousResearch/hermes-agent.git"
TAG_NAMESPACE = "refs/fork-ci/official-tags/"
STABLE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


class GuardError(RuntimeError):
    """A failed check or a tool failure; messages never include source contents."""


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GuardError("Git command could not run") from exc
    if result.returncode:
        raise GuardError(f"Git {args[0]} failed")
    return result.stdout


def resolve_baseline(repo: Path, upstream_url: str = OFFICIAL_URL) -> str:
    """Fetch official tags separately from fork tags and choose the latest ancestor."""
    _git(repo, "fetch", "--no-tags", "--prune", upstream_url,
         f"+refs/tags/v*:{TAG_NAMESPACE}v*")
    refs = _git(repo, "for-each-ref", "--format=%(refname)", TAG_NAMESPACE)
    candidates = set()
    for ref in refs.splitlines():
        match = STABLE.fullmatch(ref.removeprefix(TAG_NAMESPACE))
        if match:
            candidates.add(_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip())
    # Ancestry stays meaningful when upstream changes its version-number scheme.
    for commit in _git(repo, "rev-list", "--topo-order", "HEAD").splitlines():
        if commit in candidates:
            return commit
    raise GuardError("No official stable release is an ancestor of HEAD")


def _lint(repo: Path, head: str, path: str, added: set[int]) -> list[str]:
    source = _git(repo, "cat-file", "blob", f"{head}:{path}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--no-cache",
             "--output-format=json", "--stdin-filename", path, "-"],
            cwd=repo, input=source, text=True, encoding="utf-8", capture_output=True,
            timeout=300, check=False,
        )
        if result.returncode not in (0, 1):
            raise GuardError("Ruff check failed")
        diagnostics = json.loads(result.stdout)
        if not isinstance(diagnostics, list):
            raise ValueError("invalid diagnostics")
        return [f"ruff {item['code']} at {path!r}:{item['location']['row']}"
                for item in diagnostics if item["location"]["row"] in added]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise GuardError("Ruff check could not run or returned invalid diagnostics") from exc


def check_delta(repo: Path, base: str, head: str) -> list[str]:
    """Check exact committed base..head, trusting upstream contents and exceptions."""
    findings = []
    paths = _git(repo, "diff", "--name-only", "-z", "--no-renames", "--diff-filter=AMT",
                 base, head, "--").split("\0")
    try:
        offenders = find_offenders(repo, base, head, allowlist=allowlist_at(repo, base))
    except (GitError, OSError, subprocess.SubprocessError) as exc:
        raise GuardError("Blob-size Git check failed") from exc
    findings.extend(f"blob-size at {item.path!r}: {item.size} bytes" for item in offenders)
    for path in filter(None, paths):
        diff = _git(repo, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                    "--unified=0", base, head, "--", path)
        added = set()
        row = None
        for line in diff.splitlines():
            hunk = re.match(r"^@@ -[0-9]+(?:,[0-9]+)? \+([0-9]+)(?:,[0-9]+)? @@", line)
            if hunk:
                row = int(hunk[1])
            elif row is not None and line.startswith("+"):
                added.add(row)
                b64_exec = (re.search(r"base64\.(b64decode|decodebytes|urlsafe_b64decode)", line, re.IGNORECASE)
                            and re.search(r"exec\(|eval\(", line, re.IGNORECASE))
                encoded_process = (re.search(r"subprocess\.(Popen|call|run)\s*\(", line)
                                   and re.search(r"base64|\\x[0-9a-f]{2}|chr\(", line, re.IGNORECASE))
                if b64_exec or encoded_process:
                    findings.append(f"supply-chain encoded execution at {path!r}:{row}")
                row += 1
            elif row is not None and line.startswith(" "):
                row += 1
        if path.endswith(".py") and added:
            findings.extend(_lint(repo, head, path, added))
    return findings


def check_fork(repo: Path, upstream_url: str = OFFICIAL_URL) -> list[str]:
    """Resolve an official release, then inspect only the fork's committed delta."""
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    base = resolve_baseline(repo, upstream_url)
    print(f"fork-ci: official stable baseline {base}; candidate {head}")
    return check_delta(repo, base, head)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        findings = check_fork(Path(__file__).resolve().parents[1])
    except GuardError as exc:
        print(f"fork-ci: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"fork-ci: {finding}", file=sys.stderr)
    if findings:
        return 1
    print("fork-ci: trusted official stable fork delta passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
