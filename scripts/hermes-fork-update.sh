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
# Phases:
#   1. Sanity-check: clean tree, correct remotes, on local/runtime, venv present.
#   2. Snapshot pre-update HEAD SHA for rollback.
#   3. Fetch upstream.
#   4. Fast-forward fork's main to upstream/main (local ref only).
#   5. Rebase local/runtime onto upstream/main.
#   6. Critical-file syntax validation → auto-rollback on failure.
#   7. Clear __pycache__ to prevent stale-bytecode ImportError.
#   8. Smoke-test: ./hermes --version.
#   9. Reinstall deps IF pyproject.toml / uv.lock / requirements changed.
#  10. Restart all running hermes-gateway* + hermes-cron-* services
#      (graceful SIGUSR1 drain → fallback to systemctl restart).
#      Self-aware: if this script is running INSIDE a hermes unit (e.g. driven
#      from an agent session), that unit is skipped in the main loop and
#      restarted LAST via a detached systemd-run timer (+8s), so restarting
#      our own gateway can't SIGTERM the script and leave the remaining
#      services silently running stale pre-upgrade code.
#  11. Optional push to fork.
#
# Usage:
#   ./scripts/hermes-fork-update.sh            # full update
#   ./scripts/hermes-fork-update.sh --check    # show what would happen, no changes
#   ./scripts/hermes-fork-update.sh --push     # also push origin main + local/runtime
#   ./scripts/hermes-fork-update.sh --yes      # non-interactive
#   ./scripts/hermes-fork-update.sh --no-restart   # skip service restart phase
#   ./scripts/hermes-fork-update.sh --no-deps      # skip dep reinstall (manual)
#
# Why not `hermes update`?
#   `hermes update` hardcodes branch=main, checks out main behind your back,
#   pulls from origin/main (the fork — stale), and never replays local/runtime.
#   Even with fork-detection it strips committed local patches.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- Args ------------------------------------------------------------------
CHECK_ONLY=0
DO_PUSH=0
ASSUME_YES=0
SKIP_RESTART=0
SKIP_DEPS=0

for arg in "$@"; do
  case "$arg" in
    --check)       CHECK_ONLY=1 ;;
    --push)        DO_PUSH=1 ;;
    --yes|-y)      ASSUME_YES=1 ;;
    --no-restart)  SKIP_RESTART=1 ;;
    --no-deps)     SKIP_DEPS=1 ;;
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

# Critical files that every `hermes` invocation imports at startup.
# Matches _UPDATE_CRITICAL_FILES in hermes_cli/main.py.
CRITICAL_FILES=(
  "hermes_cli/main.py"
  "hermes_cli/config.py"
  "hermes_cli/__init__.py"
  "cli.py"
  "run_agent.py"
  "model_tools.py"
  "toolsets.py"
  "hermes_constants.py"
)

# --- 1. Pre-flight ---------------------------------------------------------
c_step "Pre-flight checks"

ORIGIN_URL="$(git remote get-url origin 2>/dev/null || echo '')"
UPSTREAM_URL="$(git remote get-url upstream 2>/dev/null || echo '')"

if [[ -z "$ORIGIN_URL" || -z "$UPSTREAM_URL" ]]; then
  c_err "Expected two remotes: 'origin' (fork) and 'upstream' (NousResearch)."
  c_err "Found: origin='$ORIGIN_URL' upstream='$UPSTREAM_URL'"
  exit 1
fi

case "$UPSTREAM_URL" in
  *NousResearch/hermes-agent*) c_ok "upstream → $UPSTREAM_URL" ;;
  *) c_warn "upstream URL is unusual: $UPSTREAM_URL (continuing)" ;;
esac

case "$ORIGIN_URL" in
  *NousResearch/hermes-agent*)
    c_err "origin points to upstream NousResearch, not your fork."
    exit 1 ;;
  *) c_ok "origin → $ORIGIN_URL" ;;
esac

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "local/runtime" ]]; then
  c_err "Expected branch 'local/runtime', currently on '$CURRENT_BRANCH'."
  c_err "Run: git checkout local/runtime"
  exit 1
fi
c_ok "on branch local/runtime"

if ! git diff --quiet || ! git diff --cached --quiet; then
  c_err "Working tree is dirty. Commit, stash, or discard changes first."
  git status --short
  exit 1
fi
c_ok "working tree clean"

# Resolve venv python (used in steps 6, 8, 9)
PYTHON=""
if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi
if [[ -n "$PYTHON" ]]; then
  c_ok "venv python: $PYTHON"
else
  c_warn "no venv found — smoke test + dep reinstall will skip"
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

LOCAL_PATCHES="$(git log --oneline upstream/main..HEAD | wc -l | tr -d ' ')"
UPSTREAM_AHEAD="$(git log --oneline HEAD..upstream/main | wc -l | tr -d ' ')"

echo
echo "  local patches on local/runtime: $LOCAL_PATCHES commits"
git log --oneline upstream/main..HEAD | sed 's/^/    /'
echo
echo "  upstream commits to pull in:    $UPSTREAM_AHEAD commits"

if [[ "$UPSTREAM_AHEAD" -eq 0 ]]; then
  c_ok "Already up to date with upstream/main. Nothing to do."
  exit 0
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  c_dim "(--check: stopping here, no changes made)"
  exit 0
fi

# --- 3. Mirror local main → upstream/main ----------------------------------
c_step "Mirroring local main to upstream/main"
if git show-ref --verify --quiet refs/heads/main; then
  git branch -f main upstream/main
else
  git branch main upstream/main
fi
c_ok "main → $UPSTREAM_SHORT"

# --- 4. Rebase local/runtime -----------------------------------------------
c_step "Rebasing local/runtime onto upstream/main"
if ! git rebase upstream/main; then
  echo
  c_warn "Rebase has conflicts. Resolve manually:"
  c_warn "  1. Edit conflicted files (see 'git status')."
  c_warn "  2. git add <files>"
  c_warn "  3. git rebase --continue"
  c_warn "  4. Re-run this script."
  c_warn ""
  c_warn "To abort and restore pre-update state:"
  c_warn "  git rebase --abort   # HEAD will be back at $PRE_SHORT"
  exit 1
fi
NEW_SHA="$(git rev-parse HEAD)"
NEW_SHORT="$(git rev-parse --short HEAD)"
c_ok "rebase complete: $PRE_SHORT → $NEW_SHORT"

# --- Rollback helper -------------------------------------------------------
ROLLBACK() {
  local reason="$1"
  c_err "$reason"
  c_err "Rolling back to $PRE_SHORT..."
  git reset --hard "$PRE_SHA"
  c_warn "Rollback complete. Investigate manually, then re-run."
  exit 1
}

# --- 5. Critical-file syntax validation ------------------------------------
c_step "Validating critical-file syntax"
if [[ -n "$PYTHON" ]]; then
  for f in "${CRITICAL_FILES[@]}"; do
    if [[ ! -f "$PROJECT_ROOT/$f" ]]; then
      continue
    fi
    if ! "$PYTHON" -m py_compile "$PROJECT_ROOT/$f" 2>&1; then
      ROLLBACK "Post-rebase syntax error in $f"
    fi
  done
  c_ok "all critical files parse cleanly"
else
  c_warn "no venv, skipped syntax check"
fi

# --- 6. Clear bytecode cache -----------------------------------------------
c_step "Clearing __pycache__ directories"
# Mirrors _clear_bytecode_cache(): excludes venv, .venv, node_modules, .git, .worktrees
REMOVED=$(find "$PROJECT_ROOT" \
  -type d \
  \( -name venv -o -name .venv -o -name node_modules -o -name .git -o -name .worktrees \) -prune \
  -o -type d -name __pycache__ -print 2>/dev/null \
  | xargs -r rm -rf -v 2>/dev/null | wc -l)
c_ok "cleared $REMOVED __pycache__ directories"

# --- 7. Smoke test ---------------------------------------------------------
c_step "Smoke test"
if [[ -n "$PYTHON" ]]; then
  if ! "$PYTHON" -c "import hermes_cli.main, run_agent" 2>&1; then
    ROLLBACK "Module import failed after rebase"
  fi
  c_ok "imports OK"

  if ! VERSION_OUT="$("$PYTHON" ./hermes --version 2>&1)"; then
    echo "$VERSION_OUT"
    ROLLBACK "./hermes --version failed"
  fi
  c_ok "hermes --version → $(echo "$VERSION_OUT" | head -1)"
else
  c_warn "no venv, skipped smoke test"
fi

# --- 8. Reinstall deps if changed ------------------------------------------
DEP_CHANGED=0
if git diff --name-only "$PRE_SHA" HEAD | grep -qE '^(pyproject\.toml|uv\.lock|requirements.*\.txt)$'; then
  DEP_CHANGED=1
fi

if [[ "$DEP_CHANGED" -eq 1 && "$SKIP_DEPS" -eq 0 && -n "$PYTHON" ]]; then
  c_step "Reinstalling Python dependencies (pyproject/uv.lock/requirements changed)"
  git diff --name-only "$PRE_SHA" HEAD | grep -E '^(pyproject|uv|requirements)' | sed 's/^/    /'
  echo
  # Prefer uv if available (matches `hermes update` behavior)
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$(dirname "$(dirname "$PYTHON")")" uv pip install -e ".[all]" 2>&1 | tail -20 || \
      c_warn "uv install hit issues (see above) — installation may be partial"
  else
    "$PYTHON" -m pip install -e ".[all]" 2>&1 | tail -20 || \
      c_warn "pip install hit issues (see above) — installation may be partial"
  fi
  c_ok "deps reinstalled"
elif [[ "$DEP_CHANGED" -eq 1 ]]; then
  c_warn "dependency files changed but --no-deps set or no venv; reinstall manually:"
  git diff --name-only "$PRE_SHA" HEAD | grep -E '^(pyproject|uv|requirements)' | sed 's/^/    /'
else
  c_ok "no dependency changes"
fi

# --- 8b. Restore the gh draft-guard shim ----------------------------------
# Defense-in-depth: all pipeline PRs must open as drafts, and published PRs
# must not be un-drafted / merged / edited by auto-workers. The guard lives in
# the venv bin ahead of the real gh; reinstall it every update in case the
# venv bin was rebuilt. Idempotent + self-skipping when no venv.
if [[ -x "$PROJECT_ROOT/scripts/install-gh-draft-guard.sh" ]]; then
  c_step "Restoring gh draft-guard shim"
  bash "$PROJECT_ROOT/scripts/install-gh-draft-guard.sh" || \
    c_warn "gh draft-guard install hit issues (see above)"
fi

# --- 9. Restart hermes services -------------------------------------------
# Which unit is THIS script running inside? If the update is driven from a
# Hermes session (agent terminal tool), the script lives in the cgroup of its
# own gateway unit. Restarting that unit kills the script mid-flight (SIGTERM
# → exit 130), so every service after it in the loop silently keeps running
# STALE pre-upgrade code while new code sits on disk. Seen in the wild
# 2026-08-08: 7 of 8 services were left on Jul-25/Aug-05 processes for ~11h.
# Fix: identify our own unit, restart it LAST, and hand that final restart to
# a detached systemd-run unit so this script survives to finish phase 10-11.
SELF_UNIT=""
if [[ -r /proc/self/cgroup ]]; then
  SELF_UNIT="$(awk -F/ '{for(i=1;i<=NF;i++) if ($i ~ /^hermes-.*\.service$/) u=$i} END{if(u) print u}' \
    /proc/self/cgroup 2>/dev/null | sed 's/\.service$//')"
fi

restart_self_detached() {
  # $1 = unit name (without .service). Restarts our OWN unit from outside the
  # script's process tree so being SIGTERMed doesn't abort the update.
  local svc="$1"
  echo "  → $svc: this is OUR OWN unit — deferring to a detached restart"
  if command -v systemd-run >/dev/null 2>&1; then
    # --on-active gives the script a few seconds to reach the summary/push
    # phase and exit cleanly before its own gateway is cycled.
    # AccuracySec=1s is REQUIRED: systemd's default timer accuracy is 1min,
    # so a bare --on-active=8s can drift ~a minute before firing. Verified
    # 2026-08-09: default drifted to +10s/unbounded, AccuracySec=1s fired at +9s.
    if systemd-run --user --quiet --collect \
         --unit="hermes-selfrestart-$$" \
         --on-active=8s \
         --timer-property=AccuracySec=1s \
         --description="deferred restart of $svc after fork-update" \
         systemctl --user restart "$svc" 2>/dev/null
    then
      c_ok "  $svc restart scheduled (+8s, detached transient timer)"
      SELF_RESTART_SCHEDULED=1
      return 0
    fi
    c_warn "  systemd-run failed — falling back to setsid"
  fi
  if command -v setsid >/dev/null 2>&1; then
    setsid --fork sh -c "sleep 8; systemctl --user restart '$svc'" \
      >/dev/null 2>&1 </dev/null &
    c_ok "  $svc restart scheduled (+8s, detached setsid)"
    SELF_RESTART_SCHEDULED=1
    return 0
  fi
  c_warn "  cannot detach — restart manually: systemctl --user restart $svc"
  return 1
}

restart_service_graceful() {
  # $1 = scope ("user" or "system"), $2 = unit name (without .service)
  local scope="$1" svc="$2"
  local scope_cmd
  if [[ "$scope" == "user" ]]; then
    scope_cmd=(systemctl --user)
  else
    scope_cmd=(sudo systemctl)
  fi

  # Get MainPID for graceful SIGUSR1 drain
  local main_pid
  main_pid="$("${scope_cmd[@]}" show "$svc" --property=MainPID --value 2>/dev/null | tr -d ' ')"
  main_pid="${main_pid:-0}"

  if [[ "$main_pid" -gt 0 ]] && [[ "$svc" == hermes-gateway* ]]; then
    # Gateway: try SIGUSR1 graceful drain (drain in-flight runs, exit 75,
    # systemd respawns).
    echo "  → $svc: draining via SIGUSR1 (PID $main_pid)..."
    kill -USR1 "$main_pid" 2>/dev/null || true
    # Wait up to 75s (60s default drain + 15s slack)
    local waited=0
    while [[ $waited -lt 75 ]]; do
      if ! kill -0 "$main_pid" 2>/dev/null; then break; fi
      sleep 1
      waited=$((waited+1))
    done
    if kill -0 "$main_pid" 2>/dev/null; then
      echo "  ⚠ $svc didn't drain in 75s — falling back to systemctl restart"
    fi
  fi

  # reset-failed + restart (handles both graceful exit + fallback paths,
  # matches `hermes update` recovery flow)
  "${scope_cmd[@]}" reset-failed "$svc" 2>/dev/null || true
  "${scope_cmd[@]}" restart "$svc" 2>&1 | head -5
  # Poll is-active for up to 15s
  local waited=0
  while [[ $waited -lt 15 ]]; do
    if [[ "$("${scope_cmd[@]}" is-active "$svc" 2>/dev/null)" == "active" ]]; then
      echo "  ✓ $svc active"
      return 0
    fi
    sleep 1
    waited=$((waited+1))
  done
  echo "  ✗ $svc failed to become active. Diagnose:"
  echo "      journalctl --user -u $svc --since '2 min ago'"
  return 1
}

if [[ "$SKIP_RESTART" -eq 1 ]]; then
  c_warn "skipping service restart (--no-restart)"
else
  c_step "Restarting hermes services"

  if [[ -n "$SELF_UNIT" ]]; then
    c_dim "  (running inside $SELF_UNIT — it will be restarted last, detached)"
  fi

  # User-scope services (gateway + cron loops)
  USER_SVCS="$(systemctl --user list-units 'hermes-*' --plain --no-legend --no-pager 2>/dev/null \
    | awk '$3 == "active" && $1 ~ /\.service$/ { sub(/\.service$/, "", $1); print $1 }')"

  if [[ -n "$USER_SVCS" ]]; then
    # NOTE: feed the loop via a here-string, NOT `echo | while`. A pipeline
    # runs the loop body in a SUBSHELL, so SELF_RESTART_SCHEDULED set inside
    # would be lost to the parent and the summary would lie.
    while read -r svc; do
      [[ -z "$svc" ]] && continue
      # Defer our own unit to the very end — restarting it here kills us.
      [[ -n "$SELF_UNIT" && "$svc" == "$SELF_UNIT" ]] && continue
      restart_service_graceful user "$svc" || true
    done <<< "$USER_SVCS"
  else
    c_dim "  (no active user-scope hermes services)"
  fi

  # System-scope services (dashboard, system gateway if installed)
  SYS_SVCS="$(systemctl list-units 'hermes-*' --plain --no-legend --no-pager 2>/dev/null \
    | awk '$3 == "active" && $1 ~ /\.service$/ { sub(/\.service$/, "", $1); print $1 }')"

  if [[ -n "$SYS_SVCS" ]]; then
    echo
    c_warn "system-scope services detected — will need sudo:"
    echo "$SYS_SVCS" | sed 's/^/    /'
    if [[ "$ASSUME_YES" -eq 1 ]]; then
      while read -r svc; do
        [[ -z "$svc" ]] && continue
        restart_service_graceful system "$svc" || true
      done <<< "$SYS_SVCS"
    else
      read -rp "  Restart system-scope services with sudo? [y/N] " yn
      if [[ "$yn" =~ ^[Yy]$ ]]; then
        while read -r svc; do
          [[ -z "$svc" ]] && continue
          restart_service_graceful system "$svc" || true
        done <<< "$SYS_SVCS"
      else
        c_warn "skipped. Restart manually: sudo systemctl restart <svc>"
      fi
    fi
  fi

  # Our own unit goes LAST, detached, so the script lives to finish.
  if [[ -n "$SELF_UNIT" ]]; then
    restart_self_detached "$SELF_UNIT" || true
  fi
fi

# --- 10. Optional push -----------------------------------------------------
if [[ "$DO_PUSH" -eq 1 ]]; then
  c_step "Pushing to fork"
  git push origin main --force-with-lease
  git push origin local/runtime --force-with-lease
  c_ok "pushed origin/main and origin/local/runtime"
else
  echo
  c_dim "To push to fork (optional):"
  c_dim "  git push origin main --force-with-lease"
  c_dim "  git push origin local/runtime --force-with-lease"
fi

# --- Summary ---------------------------------------------------------------
c_step "Summary"
echo "  upstream commits pulled in: $UPSTREAM_AHEAD"
echo "  local patches replayed:     $LOCAL_PATCHES"
echo "  pre-update HEAD:            $PRE_SHORT"
echo "  post-update HEAD:           $NEW_SHORT"
echo
if [[ "$SKIP_RESTART" -eq 1 ]]; then
  c_warn "Services NOT restarted (--no-restart). New code is on disk but every"
  c_warn "running service is still executing STALE pre-upgrade bytecode."
  c_dim  "  Restart when ready:  systemctl --user restart 'hermes-*'"
elif [[ "${SELF_RESTART_SCHEDULED:-0}" -eq 1 ]]; then
  c_ok  "All other hermes services restarted with new code."
  c_warn "$SELF_UNIT restarts in ~8s (detached) — this session will drop."
  c_dim  "  Verify after it returns:"
  c_dim  "    systemctl --user list-units 'hermes-*' --plain --no-legend"
  c_dim  "    systemctl --user show hermes-gateway -p ActiveEnterTimestamp --value"
else
  c_ok "All hermes-gateway* and hermes-cron-* services restarted with new code."
fi
c_dim "Confirm nothing is stale — every unit should show a post-upgrade start time:"
c_dim "  for s in \$(systemctl --user list-units 'hermes-*' --plain --no-legend \\"
c_dim "    | awk '{sub(/\\.service\$/,\"\",\$1); print \$1}'); do \\"
c_dim "      printf '%-34s %s\\n' \"\$s\" \"\$(systemctl --user show -p ActiveEnterTimestamp --value \$s)\"; done"
