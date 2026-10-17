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
    tests.write_text(
        '#!/usr/bin/env bash\nprintf "tests\\n" >> "$TRACE"\n'
        'printf "%s\\n" "$@" > "$RUNNER_ARGS"\n', encoding="utf-8")
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
    runner_args = tmp_path / "runner-args"
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               TRACE=str(trace), GUARD_EXIT=str(guard_exit), RUNNER_ARGS=str(runner_args))
    result = subprocess.run(helper.check_command(), cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding="utf-8", check=False, timeout=20)
    assert result.returncode == guard_exit, result.stdout + result.stderr
    stages = trace.read_text(encoding="utf-8-sig").splitlines()
    assert stages == (["prepare", "delta", "tests"] if not guard_exit else ["prepare", "delta"])
    if guard_exit:
        assert not runner_args.exists()
    else:
        selected = set(runner_args.read_text(encoding="utf-8").splitlines())
        retained_contract_suites = {
            "tests/agent/test_anthropic_prompt_cache_policy.py",
            "tests/agent/test_nous_portal_anthropic_wire.py",
            "tests/agent/test_trusted_anthropic_capabilities.py",
            "tests/agent/test_trusted_anthropic_recovery.py",
            "tests/agent/test_trusted_anthropic_accounting.py",
            "tests/agent/test_constructor_fallback_restore_accounting.py",
            "tests/agent/test_explicit_fallback_pool_restore.py",
            "tests/agent/test_fallback_restore_codex_early_reopen.py",
            "tests/hermes_cli/test_auth_codex_quota_probe.py",
            "tests/agent/test_restore_primary_pool_reselect.py",
            "tests/agent/test_reset_aware_primary_restore.py",
            "tests/agent/test_pool_revert_after_cooldown.py",
            "tests/agent/test_credential_pool_provider_boundary.py",
            "tests/agent/test_fallback_credential_isolation.py",
            "tests/agent/test_switch_model_pool_reload.py",
            "tests/plugins/platforms/test_matrix_preprocess_error_containment.py",
            "tests/agent/test_provider_fallback.py",
            "tests/agent/test_retained_named_routes.py",
            "tests/agent/test_named_route_header_lifecycle.py",
            "tests/hermes_cli/test_runtime_binding_metadata.py",
            "tests/agent/test_runtime_binding_contract.py",
            "tests/hermes_cli/test_runtime_binding_propagation.py",
            "tests/gateway/test_runtime_binding_roundtrip.py",
            "tests/tui_gateway/test_runtime_binding_restore.py",
            "tests/gateway/test_runtime_binding_integration.py",
            "tests/acp_adapter/test_runtime_binding_handoff.py",
            "tests/agent/test_runtime_binding_factory_handoffs.py",
            "tests/gateway/test_runtime_binding_api_handoff.py",
            "tests/cron/test_runtime_binding_factory.py",
            "tests/hermes_cli/test_custom_provider_extra_headers.py",
            "tests/agent/test_custom_provider_extra_headers_client.py",
            "tests/agent/test_credential_rotation_route_settings.py",
            "tests/agent/test_switch_model_reapplies_headers.py",
            "tests/agent/test_provider_attribution_headers.py",
            "tests/agent/test_env_credential_turn_refresh.py",
            "tests/agent/test_codex_usage_attribution.py",
            "tests/agent/test_codex_cloudflare_headers.py",
            "tests/agent/test_codex_service_tiers.py",
            "tests/agent/test_retained_switch_capability_contracts.py",
            "tests/gateway/test_agent_cache.py",
            "tests/hermes_cli/test_runtime_provider_resolution.py",
            "tests/hermes_cli/test_retained_provider_contracts.py",
            "tests/hermes_cli/test_trusted_capability_validation.py",
            "tests/plugins/platforms/test_matrix_sync_recovery.py",
        }
        assert retained_contract_suites <= selected
        native_factory_and_rebuild_controls = {
            "tests/acp_adapter/test_acp_dashboard_model_switch_validation.py",
            "tests/acp_adapter/test_session.py",
            "tests/acp_adapter/test_session_reasoning_config.py",
            "tests/agent/test_background_review.py",
            "tests/agent/test_background_review_cache_parity.py",
            "tests/agent/test_background_review_cost_controls.py",
            "tests/agent/test_background_review_input_budget.py",
            "tests/agent/test_background_review_list_shapes.py",
            "tests/agent/test_background_review_memory_scope.py",
            "tests/agent/test_background_review_routed_effort.py",
            "tests/agent/test_background_review_summary.py",
            "tests/agent/test_background_review_tool_call_guard.py",
            "tests/agent/test_background_review_toolset_restriction.py",
            "tests/agent/test_background_review_usage.py",
            "tests/agent/test_curator.py",
            "tests/agent/test_curator_activity.py",
            "tests/agent/test_curator_backup.py",
            "tests/agent/test_curator_classification.py",
            "tests/agent/test_curator_reports.py",
            "tests/agent/test_curator_rollback_staging.py",
            "tests/agent/test_curator_run_hygiene.py",
            "tests/agent/test_skip_background_review.py",
            "tests/cron/test_codex_execution_paths.py",
            "tests/cron/test_cron_pinned_job_fallback.py",
            "tests/cron/test_cron_provider_pin.py",
            "tests/cron/test_quota_hold.py",
            "tests/cron/test_scheduler_provider.py",
            "tests/gateway/test_active_turn_recovery.py",
            "tests/gateway/test_api_server.py",
            "tests/gateway/test_compression_in_flight_check.py",
            "tests/gateway/test_compression_interrupt_demotion.py",
            "tests/gateway/test_multiplex_api_server_routing.py",
            "tests/gateway/test_pre_agent_fallback_notice.py",
            "tests/gateway/test_runtime_provider_override_target_model.py",
            "tests/gateway/test_session_model_override_credential_pool.py",
            "tests/gateway/test_session_model_override_routing.py",
            "tests/gateway/test_session_store_stale_prune.py",
            "tests/gateway/test_turn_request_overrides.py",
            "tests/tools/test_delegate.py",
            "tests/tools/test_delegate_capability_inheritance.py",
            "tests/tools/test_delegate_external_provider.py",
            "tests/tools/test_delegate_fallback_matrix.py",
            "tests/tools/test_delegate_request_overrides.py",
            "tests/tui_gateway/test_compute_host.py",
            "tests/tui_gateway/test_compute_host_pending_model_switch.py",
            "tests/tui_gateway/test_compute_host_turn_protocol.py",
            "tests/tui_gateway/test_deferred_model_switch_confirm.py",
            "tests/tui_gateway/test_model_switch_reasoning_flag.py",
        }
        assert native_factory_and_rebuild_controls <= selected
