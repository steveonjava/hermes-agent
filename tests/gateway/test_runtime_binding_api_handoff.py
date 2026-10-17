"""API preparation boundaries only: no server, HTTP or generation."""
from unittest.mock import Mock

import pytest

from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state
from hermes_cli.config import atomic_config_replace


@pytest.fixture(autouse=True)
def named_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": "https://shared.example/v1", "api_key": "synthetic-configured"}
        for name in ("alpha", "beta")}})



@pytest.mark.parametrize("serialized", [False, True])
def test_F4_runtime_projection_and_merge_keep_authority(monkeypatch, serialized):
    from gateway.platforms.api_server import (
        _apply_runtime_agent_overrides, _resolve_request_runtime_agent_kwargs,
    )

    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    metadata = runtime_binding_state(binding) if serialized else binding
    runtime = {"provider": "custom", "requested_provider": "custom:beta", "runtime_binding": metadata,
               "api_key": "synthetic-pin", "base_url": "https://shared.example/v1"}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", Mock(return_value=runtime))
    projected = _resolve_request_runtime_agent_kwargs("custom:beta", "synthetic-model")
    target = _apply_runtime_agent_overrides({}, projected)
    assert target.get("requested_provider") == "custom:beta"
    assert target.get("runtime_binding") == metadata


@pytest.mark.parametrize("overrides", [{"runtime_binding": None}, {"runtime_binding": {}},
    {"runtime_binding": {"version": 99}}])
def test_F4_present_invalid_session_binding_rejected_without_mutation(overrides):
    from gateway.platforms.api_server import _apply_runtime_agent_overrides

    target = {"provider": "custom", "api_key": "synthetic-profile", "credential_pool": object()}
    before = dict(target)
    with pytest.raises(ValueError, match="binding"):
        _apply_runtime_agent_overrides(target, overrides)
    assert target == before


def test_F4_missing_pinned_session_key_does_not_borrow_profile():
    from gateway.platforms.api_server import _apply_runtime_agent_overrides

    target = {"provider": "custom", "api_key": "synthetic-profile", "credential_pool": object()}
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    _apply_runtime_agent_overrides(target, {"provider": "custom", "requested_provider": "custom:beta",
                                          "runtime_binding": binding})
    assert target.get("api_key") is None
    assert target.get("credential_pool") is None


@pytest.mark.parametrize("hint,policy", [({"api_key": "synthetic-route"}, "key_override"),
    ({"base_url": "https://alternate.example/v1"}, "route_override")])
def test_F4_route_intent_replaces_old_policy(hint, policy):
    from gateway.platforms.api_server import APIServerAdapter

    adapter = object.__new__(APIServerAdapter)
    adapter._session_model_override_for = lambda key: None
    adapter._recover_or_record_model = lambda model, runtime, key: model
    runtime = {"provider": "custom", "requested_provider": "custom:beta", "api_key": "synthetic-resolved",
               "credential_pool": object(), "runtime_binding": RuntimeBinding("custom", "custom:beta")}
    APIServerAdapter._select_agent_runtime(adapter, runtime, "synthetic-model", requested_model=None,
        requested_provider=None, route=hint, session_model=None, confirmed_runtime_lock=False,
        gateway_session_key=None, session_id=None)
    assert runtime["runtime_binding"].credential_policy == policy
    assert runtime["runtime_binding"].requested_provider == "custom:beta"
    assert runtime["credential_pool"] is None


@pytest.mark.parametrize("override", [{"provider": "openrouter"}, {"requested_provider": "custom:alpha"}])
def test_F4_contradiction_rejected_before_merge(override):
    from gateway.platforms.api_server import _apply_runtime_agent_overrides

    target = {"provider": "custom", "requested_provider": "custom:beta", "api_key": "synthetic"}
    before = dict(target)
    with pytest.raises(ValueError, match="binding"):
        _apply_runtime_agent_overrides(target, {"runtime_binding": RuntimeBinding("custom", "custom:beta"), **override})
    assert target == before



def test_F4_projection_to_real_sdk(monkeypatch):
    from unittest.mock import patch
    from gateway.platforms.api_server import _resolve_request_runtime_agent_kwargs, _apply_runtime_agent_overrides
    from run_agent import AIAgent
    from openai import OpenAI

    def no_network(*a, **k):
        raise AssertionError("offline constructor only")
    monkeypatch.setattr("socket.socket.connect", no_network)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=128_000),
        patch("agent.model_metadata.get_model_context_length", return_value=128_000),
    ):
        runtime = _apply_runtime_agent_overrides({}, _resolve_request_runtime_agent_kwargs("custom:beta", "synthetic-model"))
        runtime.pop("command", None)
        runtime.pop("args", None)
        agent = AIAgent(model="synthetic-model", **runtime, quiet_mode=True, skip_memory=True, skip_context_files=True)
        try:
            assert isinstance(agent.client, OpenAI)
            assert agent.requested_provider == "custom:beta"
            assert agent.runtime_binding.requested_provider == "custom:beta"
            assert agent.runtime_binding.credential_policy == "native"
        finally:
            agent.client.close()
