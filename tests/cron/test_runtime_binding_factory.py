"""Cron factory policy only; no scheduler ticks, MCP initialization or jobs run."""
from unittest.mock import Mock

import pytest

from hermes_cli.runtime_binding import RuntimeBinding


@pytest.mark.parametrize("policy", ["key_override", "route_override", "named_nonpool"])
def test_F3_detached_runtime_never_loads_independent_pool(monkeypatch, policy):
    from cron.scheduler import _load_credential_pool

    load = Mock()
    monkeypatch.setattr("agent.credential_pool.load_pool", load)
    assert _load_credential_pool({"provider": "custom", "runtime_binding":
        RuntimeBinding("custom", "custom:beta", policy)}, "synthetic-job") is None
    load.assert_not_called()


@pytest.mark.parametrize("binding", [None, RuntimeBinding("custom", "custom:beta")])
def test_F3_native_legacy_pool_reload_unchanged(monkeypatch, binding):
    from cron.scheduler import _load_credential_pool

    pool = Mock()
    pool.entries.return_value = [object()]
    load = Mock(return_value=pool)
    monkeypatch.setattr("agent.credential_pool.load_pool", load)
    assert _load_credential_pool({"provider": "custom", "runtime_binding": binding}, "synthetic-job") is pool
    load.assert_called_once_with("custom")


@pytest.mark.parametrize("policy", ["key_override", "native"])
def test_F3_constructor_receives_authority(policy):
    from cron.scheduler import _CronAgentSetup, _construct_cron_agent

    binding = RuntimeBinding("custom", "custom:beta", policy)
    setup = _CronAgentSetup(model="synthetic-model", runtime={"provider": "custom",
        "requested_provider": "custom:beta", "runtime_binding": binding})
    construct = Mock()
    _construct_cron_agent(construct, {}, {}, setup, workdir=None, session_id="synthetic", session_db=None)
    assert construct.call_args.kwargs.get("runtime_binding") is binding
    assert construct.call_args.kwargs["requested_provider"] == "custom:beta"



def test_F3_real_sdk_constructor_keeps_pin(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from unittest.mock import patch
    from cron.scheduler import _CronAgentSetup, _construct_cron_agent
    from hermes_cli.config import atomic_config_replace
    from run_agent import AIAgent
    from openai import OpenAI

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"beta": {
        "api": "https://shared.example/v1", "extra_headers": {"X-Route": "beta"}}}})
    def no_network(*a, **k):
        raise AssertionError("offline constructor only")
    monkeypatch.setattr("socket.socket.connect", no_network)
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    setup = _CronAgentSetup(model="synthetic-model", runtime={"provider": "custom",
        "requested_provider": "custom:beta", "runtime_binding": binding,
        "api_key": "synthetic-pin", "base_url": "https://shared.example/v1", "api_mode": "chat_completions"})
    with ExitStack() as stack:
        stack.enter_context(patch("model_tools.get_tool_definitions", return_value=[]))
        stack.enter_context(patch("model_tools.check_toolset_requirements", return_value={}))
        stack.enter_context(patch("agent.context_compressor.get_model_context_length", return_value=128_000))
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=128_000))
        agent = _construct_cron_agent(AIAgent, {}, {}, setup, workdir=None, session_id="synthetic", session_db=None)
        try:
            assert isinstance(agent.client, OpenAI)
            assert agent.runtime_binding is binding
            assert agent.client.default_headers.get("X-Route") == "beta"
            assert agent._try_refresh_env_client_credentials() is False
        finally:
            agent.client.close()
