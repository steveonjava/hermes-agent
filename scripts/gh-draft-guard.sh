#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# gh draft-guard shim  (Hermes pipeline safety gate)  — CANONICAL SOURCE
# ---------------------------------------------------------------------------
# This is the tracked source of the gh wrapper that gets installed at
#   <venv>/bin/gh
# ahead of the real /usr/bin/gh on every local-backend kanban worker's PATH.
#
# Install / restore it with:  scripts/install-gh-draft-guard.sh
# (the fork-update script calls that installer automatically).
#
# Enforced invariants (instruction-independent, at the tool boundary):
#   1. `gh pr create`  -> ALWAYS draft (injects --draft if omitted). No bypass.
#   2. `gh pr ready`   -> REFUSED unless GH_ALLOW_READY=1 (human action).
#   3. `gh pr merge`   -> REFUSED unless GH_ALLOW_MERGE=1 (human/maintainer).
# Everything else passes through untouched.
#
# Origin: 2026-06-09. The gateway auto-decomposer fanned a pipeline card out to
# generic `default` workers, bypassing pr-packager's --draft discipline, and one
# opened a non-draft upstream PR (#30051). This shim makes that bypass class
# impossible regardless of which profile / agent / instruction drives gh. The
# decomposer-side root cause is fixed separately (deliberate-assignment guard in
# hermes_cli/kanban_decompose.py); this shim is defense in depth.
# ---------------------------------------------------------------------------
set -euo pipefail

REAL_GH="/usr/bin/gh"
if [[ ! -x "$REAL_GH" ]]; then
    SELF="$(readlink -f "${BASH_SOURCE[0]}")"
    REAL_GH=""
    IFS=':' read -r -a _dirs <<< "$PATH"
    for d in "${_dirs[@]}"; do
        cand="$d/gh"
        if [[ -x "$cand" && "$(readlink -f "$cand")" != "$SELF" ]]; then
            REAL_GH="$cand"; break
        fi
    done
    if [[ -z "$REAL_GH" ]]; then
        echo "gh-guard: could not locate the real gh binary" >&2
        exit 127
    fi
fi

if [[ "${1:-}" == "pr" ]]; then
    sub="${2:-}"
    case "$sub" in
        create)
            has_draft=0
            for a in "$@"; do
                if [[ "$a" == "--draft" || "$a" == "-d" ]]; then has_draft=1; break; fi
            done
            if [[ "$has_draft" -eq 0 ]]; then
                echo "gh-guard: injecting --draft (all pipeline PRs open as drafts)" >&2
                shift 2
                exec "$REAL_GH" pr create --draft "$@"
            fi
            ;;
        ready)
            if [[ "${GH_ALLOW_READY:-}" != "1" ]]; then
                echo "gh-guard: REFUSED 'gh pr ready'. Flipping a PR to ready-for-review is a human action." >&2
                echo "gh-guard: a reviewer does this on GitHub after review. Set GH_ALLOW_READY=1 to override." >&2
                exit 1
            fi
            ;;
        merge)
            if [[ "${GH_ALLOW_MERGE:-}" != "1" ]]; then
                echo "gh-guard: REFUSED 'gh pr merge'. Merges are human/maintainer decisions. Set GH_ALLOW_MERGE=1 to override." >&2
                exit 1
            fi
            ;;
    esac
fi

exec "$REAL_GH" "$@"
