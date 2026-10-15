"""Behavioral checks using only synthetic, local Git repositories."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


# Build inert attack fixtures at runtime, without adding matched lines to this test's source.
def decoded(marker="SYNTHETIC_SECRET", *, operation="exec", decoder="b64decode", module="base64"):
    return f"{operation}({module}.{decoder}({marker!r}))"


def process_call(method, argument):
    return f"subprocess.{method}({argument})"


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def commit(repo, name, content):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(repo, "add", "--", name)
    git(repo, "commit", "-qm", "synthetic fixture")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path):
    official = tmp_path / "official"
    official.mkdir()
    git(official, "init", "-q", "-b", "main")
    git(official, "config", "user.name", "Fixture")
    git(official, "config", "user.email", "fixture@invalid")
    commit(official, "upstream.txt", "trusted release\n")
    git(official, "tag", "-a", "v1.2.3", "-m", "release")
    fork = tmp_path / "fork"
    subprocess.run(["git", "clone", "-q", str(official), str(fork)], check=True)
    git(fork, "config", "user.name", "Fixture")
    git(fork, "config", "user.email", "fixture@invalid")
    return official, fork


def load_api():
    path = Path(__file__).resolve().parents[2] / "scripts" / "fork_ci.py"
    assert path.exists(), "fork delta CI guard missing"
    spec = importlib.util.spec_from_file_location("fork_ci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_official_stable_ancestor_tags_define_baseline(repos):
    official, fork = repos
    api = load_api()
    base = git(official, "rev-parse", "HEAD")
    commit(fork, "local.txt", "local patch\n")
    git(fork, "tag", "-f", "v1.2.3")
    git(fork, "tag", "v99.0.0")
    for tag in ("v50.0.0rc1", "v60.0", "v70.0.0-beta", "v01.2.3"):
        git(official, "tag", tag)
    assert api.resolve_baseline(fork, str(official)) == base


def test_ancestry_order_handles_official_version_scheme_changes(repos):
    official, fork = repos
    git(official, "tag", "v2026.9.24")
    newest = commit(official, "new-release.txt", "new version scheme\n")
    git(official, "tag", "v2.0.0")
    git(fork, "fetch", "-q", str(official), "main")
    git(fork, "merge", "-q", "--ff-only", "FETCH_HEAD")
    assert load_api().resolve_baseline(fork, str(official)) == newest


def test_tag_refresh_overwrites_and_prunes_private_namespace(repos):
    official, fork = repos
    api = load_api()
    old = api.resolve_baseline(fork, str(official))
    new = commit(official, "release.txt", "new release\n")
    git(official, "tag", "-f", "v1.2.3")
    git(official, "tag", "v2.0.0")
    git(fork, "fetch", "-q", str(official), "main")
    git(fork, "merge", "-q", "--ff-only", "FETCH_HEAD")
    assert api.resolve_baseline(fork, str(official)) == new
    git(official, "tag", "-d", "v2.0.0")
    git(official, "tag", "-f", "v1.2.3", old)
    assert api.resolve_baseline(fork, str(official)) == old
    assert not git(fork, "for-each-ref", "--format=%(refname)",
                   api.TAG_NAMESPACE + "v2.0.0")


@pytest.mark.parametrize("problem", ["absent", "nonancestor", "fetch"])
def test_baseline_fails_closed(repos, problem):
    official, fork = repos
    api = load_api()
    git(official, "tag", "-d", "v1.2.3")
    if problem == "nonancestor":
        commit(official, "future.txt", "not merged\n")
        git(official, "tag", "v9.0.0")
    url = str(official / "missing") if problem == "fetch" else str(official)
    with pytest.raises(api.GuardError):
        api.resolve_baseline(fork, url)


def trusted_release(official, fork):
    commit(official, "trusted.py", "import os\n" + decoded("TRUSTED") + "\n")
    commit(official, "upstream.bin", b"\0" * 600000)
    commit(official, ".github/blob-size-allowlist.txt", "permitted.bin\n")
    git(official, "tag", "v2.0.0")
    git(fork, "fetch", "-q", str(official), "main")
    git(fork, "merge", "-q", "--ff-only", "FETCH_HEAD")


def test_trusted_upstream_findings_do_not_block_clean_fork(repos):
    official, fork = repos
    trusted_release(official, fork)
    # Editing this Python file must not report its unchanged upstream F401/F821.
    commit(fork, "trusted.py", "import os\n" + decoded("TRUSTED") + "\nvalue = 1\n")
    commit(fork, "permitted.bin", b"\0" * 700000)
    assert load_api().check_fork(fork, str(official)) == []


@pytest.mark.parametrize("line", [
    decoded(),
    decoded(operation="eval", module="BASE64", decoder="decodebytes"),
    decoded(decoder="urlsafe_b64decode"),
    decoded(operation="subprocess.run"),
    process_call("Popen", r"'\x41_SYNTHETIC_SECRET'"),
    process_call("call", "chr(65) + 'SYNTHETIC_SECRET'"),
])
def test_added_encoded_execution_is_blocked_without_content(repos, line):
    official, fork = repos
    commit(fork, "payload.txt", line + "\n")
    findings = load_api().check_fork(fork, str(official))
    assert any("supply-chain" in finding for finding in findings)
    assert "SYNTHETIC_SECRET" not in str(findings)


def test_local_allowlist_cannot_exempt_new_oversized_blob(repos):
    official, fork = repos
    commit(fork, ".github/blob-size-allowlist.txt", "local.bin\n")
    commit(fork, "local.bin", b"\0" * 600000)
    findings = load_api().check_fork(fork, str(official))
    assert any("blob-size" in finding for finding in findings)


def test_delta_includes_old_fork_changes_not_just_latest_commit(repos):
    official, fork = repos
    commit(fork, "payload.txt", decoded("OLD_LOCAL") + "\n")
    commit(fork, "clean.txt", "latest clean change\n")
    assert load_api().check_fork(fork, str(official))


def test_official_project_lint_configuration_is_respected(repos):
    official, fork = repos
    commit(official, "pyproject.toml", '[tool.ruff.lint]\nselect = ["E9", "F821"]\n')
    git(official, "tag", "v2.0.0")
    git(fork, "fetch", "-q", str(official), "main")
    git(fork, "merge", "-q", "--ff-only", "FETCH_HEAD")
    commit(fork, "local.py", "import os\n")
    assert load_api().check_fork(fork, str(official)) == []


def test_added_python_lint_is_blocked(repos):
    official, fork = repos
    commit(fork, "bad.py", "print(undefined_local_name)\n")
    findings = load_api().check_fork(fork, str(official))
    assert any("ruff F821" in finding for finding in findings)
    assert "undefined_local_name" not in str(findings)


def test_diff_failure_is_not_a_clean_delta(repos):
    official, fork = repos
    api = load_api()
    base = api.resolve_baseline(fork, str(official))
    with pytest.raises(api.GuardError, match="Git diff failed"):
        api.check_delta(fork, base, "missing-revision")


def test_ruff_failure_is_not_a_clean_delta(repos, monkeypatch):
    official, fork = repos
    api = load_api()
    commit(fork, "clean.py", "value = 1\n")
    monkeypatch.setattr(api.sys, "executable", "/nonexistent-fixture-python")
    with pytest.raises(api.GuardError, match="Ruff"):
        api.check_fork(fork, str(official))


def test_cli_rejects_baseline_override_without_fetching():
    path = Path(__file__).resolve().parents[2] / "scripts" / "fork_ci.py"
    result = subprocess.run([__import__("sys").executable, str(path), "--base", "HEAD"],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_script_help_bootstraps_blob_checker_without_pythonpath(tmp_path):
    import os
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / "fork_ci.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(path), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "trusted official stable" in result.stdout


def test_newer_nonancestor_release_does_not_replace_ancestor(repos):
    official, fork = repos
    old = git(official, "rev-parse", "HEAD")
    commit(official, "future.txt", "future release\n")
    git(official, "tag", "v10.0.0")
    assert load_api().resolve_baseline(fork, str(official)) == old


@pytest.mark.parametrize("case, expected", [("clean", 0), ("violation", 1), ("fetch", 2)])
def test_main_exit_status_with_local_real_git(repos, monkeypatch, capsys, case, expected):
    import sys

    official, fork = repos
    api = load_api()
    if case == "violation":
        commit(fork, "payload.txt", decoded("CLI_SECRET") + "\n")
    url = str(official / "missing") if case == "fetch" else str(official)
    check = api.check_fork
    monkeypatch.setattr(api, "check_fork", lambda repo: check(fork, url))
    monkeypatch.setattr(sys, "argv", ["fork_ci.py"])
    assert api.main() == expected
    output = capsys.readouterr()
    assert "CLI_SECRET" not in output.out + output.err
