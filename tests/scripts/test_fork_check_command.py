"""Exercise the shared check command with explicitly synthetic stage executables."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.platforms("linux")


@pytest.mark.parametrize("guard_exit", [0, 17], ids=["guard-passes", "guard-blocks"])
def test_shared_command_runs_delta_guard_before_regressions(tmp_path, guard_exit):
    spec = importlib.util.spec_from_file_location(
        "fork_sync_command", Path(__file__).resolve().parents[2] / "scripts/fork_sync.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / "run-in-hermes-env"
    wrapper.write_text('#!/usr/bin/env bash\nexec "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    tests = scripts / "run_tests.sh"
    tests.write_text('#!/usr/bin/env bash\nprintf "tests\\n" >> "$TRACE"\n', encoding="utf-8")
    tests.chmod(0o755)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os,pathlib,shutil,sys\n"
        "prepare=sys.argv[1:3]==['-m','pm.build_env']\n"
        "with pathlib.Path(os.environ['TRACE']).open('a',encoding='utf-8') as f:\n"
        " f.write('prepare\\n' if prepare else 'delta\\n')\n"
        "if prepare:\n"
        " p=pathlib.Path('.venv-fork-tests/bin/python');p.parent.mkdir(parents=True)\n"
        " shutil.copyfile(sys.argv[0],p);p.chmod(0o755)\n"
        "else:\n"
        " assert sys.argv[1:]==['scripts/fork_ci.py'],sys.argv\n"
        " raise SystemExit(int(os.environ['GUARD_EXIT']))\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    trace = tmp_path / "trace"
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               TRACE=str(trace), GUARD_EXIT=str(guard_exit))
    result = subprocess.run(helper.check_command(), cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding="utf-8", check=False, timeout=20)
    assert result.returncode == guard_exit, result.stdout + result.stderr
    stages = trace.read_text(encoding="utf-8-sig").splitlines()
    assert stages == (["prepare", "delta", "tests"] if not guard_exit else ["prepare", "delta"])
