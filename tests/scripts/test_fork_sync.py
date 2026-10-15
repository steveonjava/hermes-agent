import importlib.util
import json
from unittest import mock
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


class ForkSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.upstream = self.root / 'upstream'
        self.upstream.mkdir()
        git(self.upstream, 'init', '-q', '-b', 'main')
        git(self.upstream, 'config', 'user.name', 'Fixture')
        git(self.upstream, 'config', 'user.email', 'fixture@invalid')
        (self.upstream / 'core.txt').write_text('old\n')
        git(self.upstream, 'add', 'core.txt')
        git(self.upstream, 'commit', '-qm', 'base')
        self.fork = self.root / 'fork.git'
        subprocess.run(['git', 'clone', '-q', '--bare', str(self.upstream), str(self.fork)], check=True)
        local = self.root / 'local'
        subprocess.run(['git', 'clone', '-q', str(self.fork), str(local)], check=True)
        git(local, 'config', 'user.name', 'Fixture')
        git(local, 'config', 'user.email', 'fixture@invalid')
        (local / 'custom.txt').write_text('keep our patch\n')
        git(local, 'add', 'custom.txt')
        git(local, 'commit', '-qm', 'fork patch')
        git(local, 'push', '-q', 'origin', 'main')
        self.before = git(self.fork, 'rev-parse', 'main')
        (self.upstream / 'core.txt').write_text('new\n')
        git(self.upstream, 'add', 'core.txt')
        git(self.upstream, 'commit', '-qm', 'release')
        git(self.upstream, 'tag', '-a', 'v1.2.3', '-m', 'release')
        path = HERE.parents[1] / 'scripts' / 'fork_sync.py'
        self.assertTrue(path.exists(), 'fork sync mechanism missing')
        spec = importlib.util.spec_from_file_location('fork_sync', path)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)

    def run_sync(self, *, publish=False, checks=None):
        return self.api.sync(self.root / 'work', str(self.fork), str(self.upstream), 'v1.2.3',
                             checks or [sys.executable, '-c', 'import pathlib; assert pathlib.Path("custom.txt").read_text() == "keep our patch\\n"; assert pathlib.Path("core.txt").read_text() == "new\\n"'], publish)

    def test_default_check_command_covers_fork_and_updater_not_unrestricted_suite(self):
        self.assertTrue(callable(getattr(self.api, 'check_command', None)), 'bounded check command missing')
        command = self.api.check_command()
        self.assertEqual(command[:2], ['bash', '-c'])
        self.assertIn('scripts/run-in-hermes-env', command[2])
        self.assertIn('python -m pm.build_env', command[2])
        self.assertIn('--extra matrix', command[2])
        self.assertIn('CC=gcc CXX=g++', command[2])
        self.assertIn('scripts/run_tests.sh --file-retries 0 -q ', command[2])
        for test in ['tests/gateway/test_matrix.py', 'tests/plugins/platforms/test_matrix_verification.py',
                     'tests/agent/test_native_compaction_switch_capabilities.py',
                     'tests/hermes_cli/test_model_switch_custom_providers.py',
                     'tests/hermes_cli/test_update_fork_sync_identity.py', 'tests/scripts/test_fork_sync.py']:
            self.assertIn(test, command[2])

    def test_prepare_preserves_patches_and_does_not_push(self):
        result = self.run_sync()
        self.assertTrue(result['changed'])
        self.assertFalse(result['published'])
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), self.before)
        self.assertTrue((pathlib.Path(result['checkout']) / 'custom.txt').exists())

    def test_publish_retains_both_histories_and_mirrors_exact_tag(self):
        result = self.run_sync(publish=True)
        self.assertTrue(result['published'])
        head = git(self.fork, 'rev-parse', 'main')
        git(self.fork, 'merge-base', '--is-ancestor', self.before, head)
        git(self.fork, 'merge-base', '--is-ancestor', git(self.upstream, 'rev-parse', 'v1.2.3^{}'), head)
        self.assertEqual(git(self.fork, 'rev-parse', 'v1.2.3'), git(self.upstream, 'rev-parse', 'v1.2.3'))
        self.assertEqual(git(self.fork, 'show', 'main:custom.txt'), 'keep our patch')

    def test_successful_push_with_failed_readback_reports_unknown_publication(self):
        real_run = self.api.run

        def fail_readback(path, *args):
            if args[0] == 'ls-remote':
                raise self.api.SyncError('injected read-back failure')
            return real_run(path, *args)

        with mock.patch.object(self.api, 'run', side_effect=fail_readback):
            with self.assertRaisesRegex(self.api.SyncError, 'injected read-back failure'):
                self.run_sync(publish=True)

        report_path, = (self.root / 'work').glob('*-result.json')
        report = json.loads(report_path.read_text())
        checkout = pathlib.Path(report['checkout'])
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), git(checkout, 'rev-parse', 'HEAD'))
        self.assertNotEqual(git(self.fork, 'rev-parse', 'main'), self.before)
        self.assertEqual(git(self.fork, 'rev-parse', 'v1.2.3'),
                         git(self.upstream, 'rev-parse', 'v1.2.3'))
        self.assertIsNone(report['published'])
        self.assertIs(report['push_completed'], True)

    def test_main_advanced_during_checks_rejects_push_without_leaking_tag(self):
        local = self.root / 'local'
        checks = [sys.executable, '-c',
                  'import subprocess; '
                  f'subprocess.run(["git", "-C", {str(local)!r}, "commit", "--allow-empty", '
                  '"-qm", "concurrent fork patch"], check=True); '
                  f'subprocess.run(["git", "-C", {str(local)!r}, "push", "-q", '
                  '"origin", "main"], check=True)']
        with self.assertRaisesRegex(self.api.SyncError, 'rejected'):
            self.run_sync(publish=True, checks=checks)
        concurrent = git(local, 'rev-parse', 'main')
        self.assertNotEqual(concurrent, self.before)
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), concurrent)
        git(self.fork, 'merge-base', '--is-ancestor', self.before, concurrent)
        self.assertEqual(git(self.fork, 'tag', '--list', 'v1.2.3'), '')

    def test_incompatible_tag_created_during_checks_rejects_main_too(self):
        checks = [sys.executable, '-c',
                  'import subprocess; '
                  f'subprocess.run(["git", "-C", {str(self.fork)!r}, "tag", '
                  f'"v1.2.3", {self.before!r}], check=True)']
        with self.assertRaisesRegex(self.api.SyncError, 'rejected'):
            self.run_sync(publish=True, checks=checks)
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), self.before)
        self.assertEqual(git(self.fork, 'rev-parse', 'v1.2.3'), self.before)
        self.assertNotEqual(git(self.fork, 'rev-parse', 'v1.2.3'),
                            git(self.upstream, 'rev-parse', 'v1.2.3'))

    def test_second_run_is_a_noop(self):
        self.run_sync(publish=True)
        before = git(self.fork, 'rev-parse', 'main')
        result = self.run_sync(publish=True)
        self.assertFalse(result['changed'])
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), before)

    def test_failed_tests_never_publish(self):
        with self.assertRaises(self.api.SyncError):
            self.run_sync(publish=True, checks=[sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), self.before)

    def test_merge_conflict_never_publishes(self):
        (self.upstream / 'custom.txt').write_text('upstream incompatible feature\n')
        git(self.upstream, 'add', 'custom.txt')
        git(self.upstream, 'commit', '-qm', 'conflicting release')
        git(self.upstream, 'tag', '-f', 'v1.2.3')
        with self.assertRaises(self.api.SyncError):
            self.run_sync(publish=True)
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), self.before)

    def test_existing_conflicting_tag_is_not_replaced(self):
        git(self.fork, 'tag', 'v1.2.3', self.before)
        with self.assertRaises(self.api.SyncError):
            self.run_sync(publish=True)
        self.assertEqual(git(self.fork, 'rev-parse', 'main'), self.before)
        self.assertEqual(git(self.fork, 'rev-parse', 'v1.2.3'), self.before)


if __name__ == '__main__':
    unittest.main()
