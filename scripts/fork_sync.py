#!/usr/bin/env python3
"""Merge an official stable release into a fork; never update serving installs."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import urllib.request


class SyncError(RuntimeError):
    pass


def run(path, *args):
    result = subprocess.run(['git', '-C', str(path), *args], text=True, encoding='utf-8',
                            capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise SyncError(result.stderr.strip() or result.stdout.strip() or 'git failed')
    return result.stdout.strip()


def latest_release():
    request = urllib.request.Request(
        'https://api.github.com/repos/NousResearch/hermes-agent/releases/latest',
        headers={'User-Agent': 'hermes-fork-sync', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    if release.get('draft') or release.get('prerelease'):
        raise SyncError('release is not stable')
    return release['tag_name']


def sync(work_root, fork_url, upstream_url, tag, checks, publish=False):
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag):
        raise SyncError('expected an official vX.Y.Z stable tag')
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    checkout = Path(tempfile.mkdtemp(prefix='fork-sync-', dir=work_root))
    result = {'checkout': str(checkout), 'tag': tag, 'published': False, 'push_completed': False}
    try:
        run(work_root, 'clone', '--quiet', '--branch', 'main', '--single-branch', fork_url, str(checkout))
        run(checkout, 'config', 'user.name', 'Hermes Fork Sync')
        run(checkout, 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
        before = run(checkout, 'rev-parse', 'HEAD')
        run(checkout, 'remote', 'add', 'upstream', upstream_url)
        # Keep upstream's exact annotated tag object without overwriting fork tags.
        incoming = 'refs/remotes/fork-sync/upstream-tag'
        run(checkout, 'fetch', '--quiet', '--no-tags', 'upstream', 'refs/tags/' + tag + ':' + incoming)
        tag_object = run(checkout, 'rev-parse', incoming)
        upstream = run(checkout, 'rev-parse', incoming + '^{commit}')
        existing = run(checkout, 'tag', '--list', tag)
        if existing and run(checkout, 'rev-parse', 'refs/tags/' + tag) != tag_object:
            raise SyncError('fork already has a different tag object; refusing replacement')
        contained = subprocess.run(['git', '-C', str(checkout), 'merge-base', '--is-ancestor', upstream, 'HEAD'],
                                   capture_output=True, timeout=30, check=False)
        if contained.returncode not in (0, 1):
            raise SyncError('could not compare fork and stable release')
        changed = contained.returncode == 1
        if changed:
            run(checkout, 'merge', '--no-edit', upstream)
            check_home = checkout.parent / (checkout.name + '-home')
            check_home.mkdir(mode=0o700)
            check_env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(check_home),
                         'HERMES_HOME': str(check_home / '.hermes'),
                         'HERMES_RUNTIME_DIR': str(check_home / 'runtime'),
                         'LANG': 'C.UTF-8', 'TERM': 'dumb', 'PYTHONDONTWRITEBYTECODE': '1'}
            checked = subprocess.run(checks, cwd=checkout, env=check_env, timeout=3600, check=False)
            if checked.returncode:
                raise SyncError('checks failed; fork main was not published')
        after = run(checkout, 'rev-parse', 'HEAD')
        run(checkout, 'merge-base', '--is-ancestor', before, after)
        run(checkout, 'merge-base', '--is-ancestor', upstream, after)
        if not existing:
            run(checkout, 'update-ref', 'refs/tags/' + tag, tag_object, '0' * 40)
        if publish:
            # A failed push/read-back can leave the remote outcome unknown.
            result['published'] = None
            run(checkout, 'push', '--atomic', 'origin', 'HEAD:refs/heads/main', 'refs/tags/' + tag)
            result['push_completed'] = True
            remote = run(checkout, 'ls-remote', 'origin', 'refs/heads/main', 'refs/tags/' + tag)
            refs = dict(line.split('\t')[::-1] for line in remote.splitlines())
            if refs.get('refs/heads/main') != after or refs.get('refs/tags/' + tag) != tag_object:
                raise SyncError('published refs did not match; inspect remote before retrying')
        result.update(changed=changed, published=publish, before=before, after=after, upstream=upstream)
    except (SyncError, OSError, subprocess.SubprocessError) as error:
        result.update(error=str(error))
        (checkout.parent / (checkout.name + '-result.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        raise SyncError(str(error) + '\nPreserved checkout: ' + str(checkout)) from error
    (checkout.parent / (checkout.name + '-result.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def check_command():
    # Retained fork behavior plus the stock updater, not all upstream test categories.
    paths = [
        'tests/agent/test_native_compaction.py',
        'tests/agent/test_native_compaction_switch_capabilities.py',
        'tests/agent/test_primary_runtime_restore.py',
        'tests/agent/test_switch_model_context.py',
        'tests/gateway/test_matrix.py',
        'tests/gateway/test_model_switch_persistence.py',
        'tests/gateway/test_restart_notification.py',
        'tests/hermes_cli/test_kanban_db.py',
        'tests/hermes_cli/test_model_switch_custom_providers.py',
        'tests/hermes_cli/test_model_switch_native_compaction_capabilities.py',
        'tests/hermes_cli/test_resume_model_restore.py',
        'tests/plugins/platforms/test_matrix_verification.py',
        'tests/tools/test_browser_camofox_auth.py',
        'tests/tui_gateway/test_custom_provider_session_persistence.py',
        'tests/tui_gateway/test_make_agent_provider.py',
        'tests/hermes_cli/test_update_command.py',
        'tests/hermes_cli/test_update_channel.py',
        'tests/hermes_cli/test_update_fork_sync_identity.py',
        'tests/hermes_cli/test_update_upstream_prompt_noninteractive.py',
        'tests/hermes_cli/test_version_*.py',
        'tests/scripts/test_fork_sync.py',
        'tests/scripts/test_fork_ci.py',
        'tests/scripts/test_fork_check_command.py',
        'tests/ci/test_detect_changes_compare.py',
        'tests/ci/test_classify_changes.py',
        'tests/ci/test_update_ci_routing.py',
    ]
    # The explicit CI venv avoids treating PM's installed source as live home state.
    prepare = ('CC=gcc CXX=g++ python -m pm.build_env --source "$PWD" --out "$PWD/.venv-fork-tests" '
               '--group dev --group test --extra all --extra matrix')
    checks = ('"$PWD/.venv-fork-tests/bin/python" scripts/fork_ci.py && '
              'env -u __HERMES_ACTIVATED HERMES_PYTHON="$PWD/.venv-fork-tests/bin/python" '
              'scripts/run_tests.sh --file-retries 0 -q ' + ' '.join(paths))
    return ['bash', '-c', 'scripts/run-in-hermes-env bash -c ' + shlex.quote(prepare + ' && ' + checks)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', required=True, type=Path)
    parser.add_argument('--tag', help='default: latest official stable release')
    parser.add_argument('--publish', action='store_true', help='publish fork main and exact stable tag only after checks')
    args = parser.parse_args()
    try:
        result = sync(args.work_root, 'https://github.com/steveonjava/hermes-agent.git',
                      'https://github.com/NousResearch/hermes-agent.git', args.tag or latest_release(),
                      check_command(), args.publish)
    except (SyncError, OSError, ValueError) as error:
        print(str(error))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
