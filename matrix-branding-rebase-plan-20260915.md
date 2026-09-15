# Hermes Matrix Branding Parity - Revised Source-Only Plan

## Scope and guardrails

- Reconstruct the reviewed Matrix self-profile feature on freshly fetched `steveonjava/hermes-agent` `main`, preserving its current plugin adapter layout.
- Keep the feature opt-in through `matrix.self_profile` in `config.yaml`; do not add environment-variable configuration.
- Limit behavior to the authenticated bot account's global display name and avatar. Do not mutate Matrix rooms, room state, membership, encryption, invitations, live CT303 services, or runtime profile state.
- Keep the prior dirty upstream-based worktree unchanged as evidence only. Use candidate `ec4126580726bb83e3d7116bb5e72cab24cca6ac` only to recover intent, not as a merge base.

## Prior-art and reconciliation

- Inspect current `plugins/platforms/matrix/adapter.py`, `tests/gateway/test_matrix.py`, and `website/docs/user-guide/messaging/matrix.md`.
- Recheck upstream and fork issue, PR, commit, code, and history searches for Matrix self-profile, room branding, avatar, topic, and display-name behavior.
- Preserve current `main` conventions and all newer Matrix adapter behavior; manually recreate only compatible self-profile logic.

## TDD implementation

- Add behavioral tests first for YAML-to-adapter propagation, independent configuration-field validation, changed-value reconciliation, failure isolation, and absence of room calls.
- Confirm new tests fail on the clean baseline because the feature is absent.
- Implement minimal adapter behavior: validate explicit `self_profile`, retain it through the YAML bridge, and reconcile only changed global profile fields after authentication.
- Validate MXC avatar shape conservatively: one non-empty server and media-id component; reject whitespace, query, and fragment characters.
- Document the explicit opt-in, valid fields, non-effects, and non-blocking failure behavior.

## Verification and publication

- Run the complete Matrix gateway test file; run Ruff lint, formatter assessment, type checks, docs checks, diff checks, and secret scans. Record environmental or pre-existing-check blockers precisely.
- Commit the reconciled source, tests, documentation, non-blocking background profile sync, and strict MXC server validation on one new immutable fork branch; push without force and do not merge `main`.
- Attach this exact plan as a native Matrix `m.file`, download it from Matrix, and compare its SHA-256 to the local artifact.
- Post a task- and hostname-specific source-only status update to `!xwKWZbbmZarPHKDEeU:matrix.browsecode.org`.
- Request two fresh independent reviews against the exact pushed commit. Stop before any integration, deployment, gateway restart, profile mutation, room operation, canary, private-KB lifecycle change, or CT303 runtime change.
