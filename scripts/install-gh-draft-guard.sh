#!/usr/bin/env bash
# Install/restore the gh draft-guard shim into the active venv's bin, ahead of
# the real gh on PATH. Idempotent. Safe to run on every fork update.
#
# Resolves the venv the same way hermes-fork-update.sh does.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC="$SCRIPT_DIR/gh-draft-guard.sh"

if [[ ! -f "$SRC" ]]; then
    echo "install-gh-draft-guard: source $SRC missing" >&2
    exit 1
fi

# Resolve venv bin dir.
if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    VENV_BIN="$PROJECT_ROOT/venv/bin"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    VENV_BIN="$PROJECT_ROOT/.venv/bin"
else
    echo "install-gh-draft-guard: no venv found under $PROJECT_ROOT — skipping" >&2
    exit 0
fi

DEST="$VENV_BIN/gh"

# Never shadow ourselves: if the real gh somehow lives in the venv bin, bail.
if [[ -e "$DEST" ]] && ! grep -q "gh-guard" "$DEST" 2>/dev/null; then
    # An existing non-guard gh in the venv bin would be the real binary; do not
    # clobber it blindly.
    echo "install-gh-draft-guard: $DEST exists and is not the guard — leaving it alone" >&2
    exit 0
fi

install -m 0755 "$SRC" "$DEST"
echo "install-gh-draft-guard: installed guard at $DEST"
