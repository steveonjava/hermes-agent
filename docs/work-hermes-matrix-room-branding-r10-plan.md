# Hermes Matrix Branding R10 Correction Plan

Task `work-hermes-matrix-room-branding` repairs raw self-profile exception logging in the Hermes Matrix gateway on `hermes-homelab` (CT303), source only.

- Clean branch/worktree: `work/work-hermes-matrix-room-branding/worker-1a-r10`, `/home/hermes/.hermes/profiles/worker-1a/home/source-worktrees/work-hermes-matrix-room-branding-r10`.
- Fresh fork base: `cf28048bbab0c57a40f9f8dfd77ac2f4ed29a8b9` from `steveonjava/hermes-agent` main.
- No R8/R9 or upstream-based worktree will be read for edits, merged, or changed.

## Confirmed Gap

`_sync_self_profile` logs raw exceptions in both display-name and avatar failure paths. Exceptions can contain configured values and control characters. The repair removes only exception interpolation while preserving generic field-specific warnings, independent field execution, bounded deferred sync, E2EE/key-share ordering, changed-only writes, cancellation, and strict MXC validation.

## Strict TDD

1. Add separate display-name and avatar tests with real runtime `\x1b[31m` values and configured-value sentinels; retain the current timeout and independent-field-failure tests unchanged.
2. Run each focused test against the exact base and observe RED from leaked values/control data.
3. Remove raw exception interpolation from exactly the two warning calls.
4. Rerun focused tests for GREEN and the complete Matrix gateway test file.

## Gates And Publication

Run Matrix tests, Ruff check/format, adapter type check with baseline comparison, website docs check/build, `git diff --check`, and scoped secret scan. Inspect the Matrix docs and leave them unchanged only if they remain accurate. Commit the repair and this plan, publish exactly one non-force R10 branch, read its remote SHA back, attach this exact plan as native `m.file` with byte/SHA-256 verification, post evidence, and stop for two independent reviews.

No integration, deployment, restart, live Matrix profile/room action, canary, private-KB lifecycle action, or CT303 runtime change is authorized.
