"""Replay the real action with synthetic, endpoint-aware GitHub API fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import hermes_yaml
import pytest


_REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.platforms("linux")
_COMPARE = "repos/fixture/fork/compare/" + "a" * 40 + "..." + "b" * 40
_PR = "repos/fixture/fork/pulls/7"


def _replay(tmp_path, comparison, *, labels=(), direct=False, failures=0):
    action = hermes_yaml.safe_load(
        (_REPO / ".github/actions/detect-changes/action.yml").read_text(encoding="utf-8-sig")
    )
    # Recovery deliberately narrows the lanes: it must not override fail-open.
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps({
        _COMPARE: comparison,
        _PR + "/files": [{"filename": "website/docusaurus.config.ts"}],
        _PR: {"labels": [{"name": name} for name in labels]},
    }), encoding="utf-8")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7, "labels": []}}), encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    sleeps = tmp_path / "sleeps"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,subprocess,sys\n"
        "calls=pathlib.Path(os.environ['FIXTURE_CALLS'])\n"
        "previous=calls.read_text(encoding='utf-8-sig').splitlines() if calls.exists() else []\n"
        "with calls.open('a',encoding='utf-8') as f:\n"
        " f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "endpoint=next(arg for arg in sys.argv if arg.startswith('repos/'))\n"
        "if '/compare/' in endpoint and len(previous)<int(os.environ['FIXTURE_FAILURES']):\n"
        " raise SystemExit(1)\n"
        "responses=json.loads(pathlib.Path(os.environ['FIXTURE_RESPONSES']).read_text(encoding='utf-8-sig'))\n"
        "data=json.dumps(responses[endpoint])\n"
        "if '--jq' in sys.argv:\n"
        " result=subprocess.run(['jq','-r',sys.argv[sys.argv.index('--jq')+1]],input=data,"
        "text=True,encoding='utf-8',check=False)\n"
        " raise SystemExit(result.returncode)\n"
        "print(data)\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    (bindir / "python3").symlink_to(sys.executable)
    sleep = bindir / "sleep"
    sleep.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$FIXTURE_SLEEPS"\n', encoding="utf-8")
    sleep.chmod(0o755)
    output = tmp_path / "outputs"
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               EVENT_NAME="pull_request", REPO="fixture/fork", BASE_SHA="a" * 40,
               HEAD_SHA="b" * 40, GH_TOKEN="fixture-not-a-credential",
               GITHUB_EVENT_PATH=str(event), CI_CLASSIFY_FORCE_ALL="1",
               FIXTURE_RESPONSES=str(responses), FIXTURE_CALLS=str(calls),
               FIXTURE_FAILURES=str(failures), FIXTURE_SLEEPS=str(sleeps),
               GITHUB_OUTPUT=str(output))
    if direct:
        env.pop("CI_CLASSIFY_FORCE_ALL")
    command = ([sys.executable, "scripts/ci/classify_changes.py"] if direct else
               ["bash", "-c", action["runs"]["steps"][0]["run"]])
    result = subprocess.run(command, cwd=_REPO, input="", env=env,
                            text=True, encoding="utf-8", capture_output=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    lanes = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8-sig").splitlines())
    requests = [json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()]
    compares = [call for call in requests if _COMPARE in call]
    assert all("--paginate" not in call for call in compares)
    assert len(compares) == (0 if direct else min(failures + 1, 3))
    delays = sleeps.read_text(encoding="utf-8-sig").splitlines() if sleeps.exists() else []
    assert delays == ["10"] * min(failures, 2)
    return lanes, requests, result


def _comparison(count):
    return {"total_commits": 10000,
            "files": [{"filename": f"website/static/{i}.svg"} for i in range(count)]}


def test_capped_file_list_runs_every_lane_instead_of_recovering_partial_diff(tmp_path):
    lanes, calls, result = _replay(tmp_path, _comparison(300))
    assert lanes and all(value == "true" for value in lanes.values()), result.stderr
    assert len(calls) == 1  # Neither PR-files recovery nor labels can narrow this result.
    assert "300-file" in result.stdout


@pytest.mark.parametrize("comparison", [{}, {"files": None}, {"files": []},
                                        {"files": [{"filename": "  "}]}],
                         ids=["missing", "null", "empty", "blank"])
def test_missing_comparison_runs_every_lane_without_recovery(tmp_path, comparison):
    lanes, calls, result = _replay(tmp_path, comparison)
    assert lanes and all(value == "true" for value in lanes.values()), result.stderr
    assert len(calls) == 1


@pytest.mark.parametrize("count", [1, 299], ids=["small", "near-cap"])
def test_uncapped_comparison_resets_force_signal_and_keeps_normal_routing(tmp_path, count):
    lanes, calls, result = _replay(tmp_path, _comparison(count))
    assert lanes["site"] == "true"
    assert lanes["python"] == lanes["frontend"] == lanes["e2e"] == "false"
    assert len(calls) == 2
    assert _PR in calls[1]
    assert "300-file" not in result.stdout


def test_small_comparison_keeps_live_label_routing(tmp_path):
    lanes, calls, _ = _replay(tmp_path, _comparison(1), labels=["run-e2e"])
    assert lanes["site"] == lanes["e2e"] == "true"
    assert lanes["frontend"] == lanes["python"] == "false"
    assert len(calls) == 2
    assert _PR in calls[1]


def test_direct_classifier_keeps_pr_files_recovery(tmp_path):
    lanes, calls, result = _replay(tmp_path, {}, direct=True)
    assert lanes["site"] == "true"
    assert lanes["python"] == lanes["frontend"] == "false"
    assert _PR + "/files" in calls[0]
    assert _PR in calls[1]
    assert "recovered 1 path(s)" in result.stderr


@pytest.mark.parametrize("failures", [2, 3], ids=["third-attempt-success", "exhausted"])
def test_compare_preserves_three_attempts_and_backoff(tmp_path, failures):
    lanes, calls, _ = _replay(tmp_path, _comparison(1), failures=failures)
    if failures == 3:
        assert lanes and all(value == "true" for value in lanes.values())
        assert len(calls) == 3
    else:
        assert lanes["python"] == "false"
        assert len(calls) == 4
