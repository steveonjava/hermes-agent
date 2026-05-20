#!/usr/bin/env bash
# hermes-fork-update.sh
#
# Update a Hermes checkout that runs from a long-lived "local/runtime" branch
# off your own fork, carrying local patches on top of upstream main.
#
# This is the SAFE replacement for `hermes update` when you have the layout:
#
#   origin    → steveonjava/hermes-agent  (your fork)
#   upstream  → NousResearch/hermes-agent (upstream, read-only)
#   branch    → local/runtime  (commits = local patches not yet merged upstream)
#
# What it does, in order:
#   1. Sanity-check: clean working tree, correct remotes, currently on
#      local/runtime, venv present.
#   2. Snapshot the pre-update SHA so we can roll back.
#   3. Fetch upstream.
#   4. Fast-forward fork's main to upstream/main (locally; optional push).
#   5. Rebase local/runtime onto upstream/main.
#   6. Drop into a conflict-resolution shell if rebase has conflicts.
#   7. Smoke-test: import critical modules, run `./hermes --version`.
#   8. Optionally reinstall deps if pyproject.toml changed.
#   9. Print summary + push instructions.
#
# What it deliberately does NOT do (vs. `hermes update`):
#   - Doesn't touch the gateway or running agents.
#   - Doesn't stash uncommitted changes invisibly (refuses to run dirty).
#   - Doesn't checkout main behind your back.
#   - Doesn't auto-push (no creds assumed on this host).
#
# Why not just use `hermes update`?
#   `hermes update` hardcodes `branch=main`, switches you off local/runtime,
#   pulls origin/main, and never replays your local commits.  Even with
#   fork-detection it assumes the fork is a friendly mirror of upstream —
#   our local/runtime model carries real patches it would silently strip.
#
# Usage:
#   ./scripts/hermes-fork-update.sh            # interactive, default
#   ./scripts/hermes-fork-update.sh --check    # show what would happen, don't change anything
#   ./scripts/hermes-fork-update.sh --push     # also push origin main + local/runtime
#   ./scripts/hermes-fork-update.sh --yes      # non-interactive (no conflict shell)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- Defaults / args -------------------------------------------------------
CHECK_ONLY=0
DO_PUSH=0
ASSUME_YES=0

for arg in "$@"; do
  case "$arg" in
    --check)  CHECK_ONLY=1 ;;
    --push)   DO_PUSH=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --help|-h)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "✗ Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# --- Pretty printing -------------------------------------------------------
c_dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
c_ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
c_warn() { printf '\033[33m⚠\033[0m %s\n' "$*"; }
c_err()  { printf '\033[31m✗\033[0m %s\n' "$*" >&2; }
c_step() { printf '\n\033[1;36m→\033[0m %s\n' "$*"; }

# --- 1. Pre-flight sanity checks -------------------------------------------
c_step "Pre-flight checks"

# Remotes
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || echo '')"
UPSTREAM_URL="$(git remote get-url upstream 2>/dev/null || echo '')"

if [[ -z "$ORIGIN_URL" || -z "$UPSTREAM_URL" ]]; then
  c_err "Expected two remotes: 'origin' (fork) and 'upstream' (NousResearch)."
  c_err "Found: origin='$ORIGIN_URL' upstream='$UPSTREAM_URL'"
  c_err "Set them up with:"
  c_err "  git remote add upstream https://github.com/NousResearch/hermes-agent.git"
  c_err "  git remote set-url origin https://github.com/<you>/hermes-agent.git"
  exit 1
fi

case "$UPSTREAM_URL" in
  *NousResearch/hermes-agent*) c_ok "upstream → $UPSTREAM_URL" ;;
  *) c_warn "upstream URL is unusual: $UPSTREAM_URL (continuing)" ;;
esac

case "$ORIGIN_URL" in
  *NousResearch/hermes-agent*)
    c_err "origin points to upstream NousResearch, not your fork."
    c_err "  This script is for fork-based runtimes. Repoint origin first."
    exit 1
    ;;
  *) c_ok "origin → $ORIGIN_URL" ;;
esac

# Branch
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "local/runtime" ]]; then
  c_err "Expected to be on branch 'local/runtime', currently on '$CURRENT_BRANCH'."
  c_err "Run: git checkout local/runtime"
  exit 1
fi
c_ok "on branch local/runtime"

# Clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
  c_err "Working tree is dirty. Commit, stash, or discard changes before updating."
  c_err "  git status"
  git status --short
  exit 1
fi
c_ok "working tree clean"

# Venv present
if [[ ! -d "$PROJECT_ROOT/venv" && ! -d "$PROJECT_ROOT/.venv" ]]; then
  c_warn "No venv/ or .venv/ found — smoke test will skip."
fi

PRE_SHA="$(git rev-parse HEAD)"
PRE_SHORT="$(git rev-parse --short HEAD)"
c_ok "current HEAD: $PRE_SHORT"

# --- 2. Fetch upstream -----------------------------------------------------
c_step "Fetching upstream"
git fetch upstream --quiet
UPSTREAM_SHA="$(git rev-parse upstream/main)"
UPSTREAM_SHORT="$(git rev-parse --short upstream/main)"
c_ok "upstream/main: $UPSTREAM_SHORT"

# --- 3. Compute what will change -------------------------------------------
LOCAL_PATCHES="$(git log --oneline upstream/main..HEAD | wc -l | tr -d ' ')"
UPSTREAM_AHEAD="$(git log --oneline HEAD..upstream/main | wc -l | tr -d ' ')"
MERGE_BASE="$(git merge-base HEAD upstream/main)"
MERGE_BASE_SHORT="$(git rev-parse --short "$MERGE_BASE")"

echo
echo "  local patches on local/runtime: $LOCAL_PATCHES commits"
git log --oneline upstream/main..HEAD | sed 's/^/    /'
echo
echo "  upstream commits to pull in:    $UPSTREAM_AHEAD commits"
if [[ "$UPSTREAM_AHEAD" -gt 0 ]]; then
  echo "  merge base: $MERGE_BASE_SHORT"
fi

if [[ "$UPSTREAM_AHEAD" -eq 0 ]]; then
  c_ok "Already up to date with upstream/main. Nothing to do."
  exit 0
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  c_dim "(--check: stopping here, no changes made)"
  exit 0
fi

# --- 4. Update fork's main to upstream/main (local only) -------------------
c_step "Mirroring fork main to upstream/main (local refs)"
# Update the local 'main' branch ref without checkout (safe even from
# local/runtime). If main doesn't exist locally, create it.
if git show-ref --verify --quiet refs/heads/main; then
  git branch -f main upstream/main
else
  git branch main upstream/main
fi
c_ok "main → $UPSTREAM_SHORT (matches upstream)"

# --- 5. Rebase local/runtime onto upstream/main ----------------------------
c_step "Rebasing local/runtime onto upstream/main"
if ! git rebase upstream/main; then
  echo
  c_warn "Rebase has conflicts. Resolve manually:"
  c_warn "  1. Edit conflicted files (see 'git status')."
  c_warn "  2. git add <files>"
  c_warn "  3. git rebase --continue"
  c_warn "  4. Re-run this script with --check to verify."
  c_warn ""
  c_warn "To abort and restore pre-update state:"
  c_warn "  git rebase --abort"
  c_warn "  # then verify HEAD == $PRE_SHORT"
  exit 1
fi
NEW_SHA="$(git rev-parse HEAD)"
NEW_SHORT="$(git rev-parse --short HEAD)"
c_ok "rebase complete: $PRE_SHORT → $NEW_SHORT"

# --- 6. Smoke test ---------------------------------------------------------
c_step "Smoke test"

PYTHON=""
if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi

ROLLBACK() {
  c_err "Smoke test failed. Rolling back to $PRE_SHORT..."
  git reset --hard "$PRE_SHA"
  c_warn "Rolled back. Investigate manually, then re-run."
  exit 1
}

if [[ -n "$PYTHON" ]]; then
  if ! "$PYTHON" -c "import hermes_cli.main, run_agent" 2>&1; then
    ROLLBACK
  fi
  c_ok "imports OK"

  if ! "$PYTHON" ./hermes --version >/dev/null 2>&1; then
    # Show output for diagnosis
    "$PYTHON" ./hermes --version || true
    ROLLBACK
  fi
  VERSION_LINE="$("$PYTHON" ./hermes --version 2>/dev/null | head -1)"
  c_ok "hermes --version → $VERSION_LINE"
else
  c_warn "no venv found, skipped python smoke test"
fi

# --- 7. Check if deps need reinstalling ------------------------------------
if git diff --name-only "$PRE_SHA" HEAD | grep -qE '^(pyproject\.toml|uv\.lock|requirements.*\.txt)$'; then
  c_warn "Dependency files changed in this update:"
  git diff --name-only "$PRE_SHA" HEAD | grep -E '^(pyproject|uv|requirements)' | sed 's/^/    /'
  echo
  echo "  Reinstall deps with one of:"
  echo "    uv pip install -e .[all]"
  echo "    venv/bin/pip install -e .[all]"
  echo
else
  c_ok "no dependency changes"
fi

# --- 8. Optional push ------------------------------------------------------
if [[ "$DO_PUSH" -eq 1 ]]; then
  c_step "Pushing to fork"
  git push origin main --force-with-lease
  git push origin local/runtime --force-with-lease
  c_ok "pushed origin/main and origin/local/runtime"
else
  echo
  c_dim "To push the updated branches to your fork (optional):"
  c_dim "  git push origin main --force-with-lease"
  c_dim "  git push origin local/runtime --force-with-lease"
fi

# --- 9. Summary ------------------------------------------------------------
c_step "Summary"
echo "  upstream commits pulled in: $UPSTREAM_AHEAD"
echo "  local patches replayed:     $LOCAL_PATCHES"
echo "  pre-update HEAD:            $PRE_SHORT"
echo "  post-update HEAD:           $NEW_SHORT"
echo
c_dim "Restart any running hermes processes (gateway, cron, etc.) to pick up the new code."
