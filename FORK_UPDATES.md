# Fork maintenance and ordinary Hermes updates

This fork keeps local product patches in Git. It does not require custom runtime
archives, privileged source ownership, or a separate application updater.

## Daily source sync

Run the helper once daily from the existing maintenance controller, or manually.
It resolves the latest published official stable release, clones fork `main` into
an isolated directory, and merges that release while retaining fork commits.
A changed candidate prepares the standard PM test environment and runs the
canonical fork-customization and native-updater regression checks. Only success permits a normal, atomic push of `main` plus
the exact upstream release tag. There is no force push, patch reset, or automatic
conflict resolution. A conflict, test failure, tag collision, or racing main
update stops publication and preserves the checkout for diagnosis. A no-op
release does not rerun the suite. This helper updates GitHub only, never hosts.

Prepare manually (does not push):

```sh
python3 scripts/fork_sync.py --work-root "${XDG_CACHE_HOME:-$HOME/.cache}/hermes-fork-sync"
```

Use the existing maintainer Git credential. The default GitHub Actions token
cannot update `.github/workflows/`, so a second Actions writer/token is deliberately
not required. Scheduling and host deployment stay with the existing controller.
Do not squash/rebase upstream integration merges: retain their ancestry so future
merges recognize releases already included.

## Ordinary installations

Install as the account that will run Hermes. Use the stock installer with
`HERMES_REPO_URL=https://github.com/steveonjava/hermes-agent.git`; the normal source
checkout, PM-managed dependencies and command launchers remain user-owned.
Keep `origin` pointing at this fork. Upstream integration belongs in the source
sync job, not in each serving checkout, so instances do not need an `upstream`
remote or GitHub write credentials. Forks normally track their own `main`.

After the one-time source-install migration, routine operation is standard:

```sh
hermes update --plan
hermes update --yes
```

The built-in updater owns dependency preparation, normal backups/config
migration, product builds and restart handling. Do not copy application patches
into individual installations. Push product changes through this fork instead.
For fleet restarts, invoke the command outside the gateway being updated; do not
make a gateway kill its own control session. A currently running installation
that points at an old custom runtime still needs a separately verified one-time
migration; updating a different checkout does not update that running process.
