# work-hermes-matrix-room-branding R6 Source-Only Repair Plan

## Task And Authority

Task `work-hermes-matrix-room-branding` repairs observable Matrix self-profile behavior in the Hermes Agent source for `hermes-homelab` (CT303). This R6 plan supersedes the earlier workspace-bound plan because the successor workspace disappeared before any production implementation. This is source-only: no deployment, gateway restart, profile or room mutation, Matrix media, canary, or CT303 runtime change is permitted.

- Clean R6 branch: `work/work-hermes-matrix-room-branding/worker-1a-r6`.
- Clean R6 workspace: `/home/hermes/.hermes/profiles/worker-1a/home/source-worktrees/work-hermes-matrix-room-branding-r6`.
- Authoritative fork remote: `https://github.com/steveonjava/hermes-agent.git` (`origin`).
- `git ls-remote` verified remote `main` and fresh cloned base both equal `c9655bfa704ff1ecf21d9c27bc3bfbabf92c2d79`.
- Prior candidate `b883a8516b1de1d80aa5b7f55f43d0e74532d8cf` is not a current remote authority; it is research input only.

## Scope

Reconstruct only the smallest opt-in Matrix global self-profile synchronizer necessary to preserve the candidate's observable contracts:

- explicit YAML-only `matrix.self_profile` configuration; no behavioral environment variable;
- display-name and opaque `mxc://` avatar self-profile updates only when the current profile differs;
- no HTTP/path avatar fetching or uploading;
- non-blocking startup: profile work runs after connection is live and is bounded/cancellable;
- E2EE setup and initial key-share ordering remain unchanged before profile work begins;
- strict MXC authority validation accepts hostnames, IPv4, valid `host:port`, and bracketed IPv6 with optional port, while rejecting malformed, ambiguous, userinfo, query/fragment, path-in-authority, whitespace, and invalid-port forms;
- profile-sync failures are redacted and never prevent connection success.

No room state, membership, encryption setting, authentication state, or media is changed by the source implementation.

## Strict TDD

For each vertical behavior slice, write one focused test first, run it to observe the expected failure, write the smallest production implementation, rerun the focused test, then proceed:

1. Explicit `self_profile` parsing is opt-in and rejects absent/invalid values.
2. MXC authority validation accepts the required syntax and rejects malformed/ambiguous inputs.
3. Unchanged display name/avatar produces no setter call; changed values invoke only the corresponding self-profile setter.
4. Profile work is scheduled only after authentication, E2EE setup, initial sync, and key sharing; it cannot delay `connect()` success.
5. Bounded profile failure is redacted, isolated, cancellable during disconnect, and does not change a successful connection result.
6. YAML bridge passes only `self_profile` into Matrix adapter configuration without environment fallback.

Tests must use real adapter code with faked Matrix transport and a temporary `HERMES_HOME` where configuration propagation is exercised.

## Verification And Reviews

1. Run every focused RED/GREEN test command, then the complete Matrix gateway test file, relevant config/plugin tests, formatting/lint/type gates available in the repository, and `git diff --check`.
2. Verify the final branch still descends from the measured current `origin/main`; re-read the remote ref before publishing.
3. Commit explicit paths only, push only this R6 successor branch, then verify the remote successor ref equals the local commit before calling it immutable.
4. Attach this exact plan Markdown in the task thread, read the `m.file` event back, download it, and verify byte equality and SHA-256.
5. Request two fresh independent commit-bound reviews in this thread. Reviews are limited to observable self-profile correctness, startup non-blocking behavior, E2EE ordering, MXC syntax, redacted failures, tests/docs, and preservation of newer main. Theoretical risks without a reachable path are non-blocking.
6. Repair only reproduced review defects in a new immutable successor, repeat gates and reviews as required, then emit terminal evidence. No integration or deployment is part of this task.

## Rollback

No runtime state changes occur. Source rollback is controller non-integration of the R6 branch, or a later revert on that branch if a reproduced defect requires it.
