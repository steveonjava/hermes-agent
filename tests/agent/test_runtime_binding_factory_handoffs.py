"""Offline real constructor handoffs; no review/delegate generation is performed."""
from unittest.mock import patch

import pytest

from hermes_cli.config import atomic_config_replace
from hermes_cli.runtime_binding import RuntimeBinding
from tests.agent import test_retained_named_routes as route_fixtures


offline_context_metadata = route_fixtures.offline_context_metadata


@pytest.fixture
def routes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": "https://shared.example/v1", "api_key": "synthetic-configured",
               "transport": "chat_completions", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})


@pytest.fixture
def real_constructor():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=128_000),
    ):
        yield AIAgent


def make_parent(real_constructor, policy="key_override"):
    return real_constructor(
        model="synthetic-model", provider="custom", requested_provider="custom:beta",
        runtime_binding=RuntimeBinding("custom", "custom:beta", policy),
        api_key="synthetic-pinned", base_url="https://shared.example/v1",
        api_mode="chat_completions", quiet_mode=True, skip_memory=True, skip_context_files=True,
    )


@pytest.mark.parametrize("status", [401, 429])
def test_F1_same_parent_real_fork_keeps_selected_pin(routes, real_constructor, status):
    from agent.background_review import build_cache_parity_fork
    from openai import OpenAI

    parent = make_parent(real_constructor)
    fork = None
    try:
        fork, runtime, routed = build_cache_parity_fork(parent, {}, max_iterations=1)
        assert isinstance(fork.client, OpenAI)
        assert fork.requested_provider == "custom:beta"
        assert fork.runtime_binding is parent.runtime_binding
        assert runtime["runtime_binding"] is parent.runtime_binding
        assert fork.client.default_headers.get("X-Route") == "beta"
        assert fork._recover_with_credential_pool(status_code=status, has_retried_429=True)[0] is False
        assert fork._try_refresh_env_client_credentials() is False
        assert fork.api_key == parent.api_key
        assert not routed
        assert fork._persist_disabled and fork._session_db is None
        assert fork.session_id == parent.session_id
        assert fork.tools == parent.tools
        assert fork._cached_system_prompt == parent._cached_system_prompt
    finally:
        if fork is not None:
            fork.client.close()
        parent.client.close()


@pytest.mark.parametrize("override,policy", [({}, "native"), ({"api_key": "synthetic-slot"}, "key_override"),
    ({"base_url": "https://alternate.example/v1"}, "route_override")])
def test_F1_independent_route_owns_binding(routes, real_constructor, override, policy):
    from agent.background_review import build_cache_parity_fork

    parent = make_parent(real_constructor)
    fork = None
    try:
        fork, runtime, routed = build_cache_parity_fork(parent, {
            "provider": "custom:alpha", "model": "other-model", **override}, max_iterations=1)
        assert routed
        assert fork.requested_provider == "custom:alpha"
        assert fork.runtime_binding.credential_policy == policy
        assert fork.runtime_binding is not parent.runtime_binding
        assert runtime["runtime_binding"] is fork.runtime_binding
        assert fork.client.default_headers.get("X-Route") == "alpha"
    finally:
        if fork is not None:
            fork.client.close()
        parent.client.close()


@pytest.mark.parametrize("slot,policy", [(False, "native"), (True, "key_override")])
def test_F2_curator_real_constructor_projection(routes, real_constructor, monkeypatch, slot, policy):
    from agent import curator

    config = {"model": {"provider": "custom:beta", "default": "synthetic-model"}}
    if slot:
        config["auxiliary"] = {"curator": {"provider": "custom:beta", "model": "synthetic-model",
                                             "api_key": "synthetic-slot"}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    captured = []
    sdk_headers = []
    def construct(**kwargs):
        agent = real_constructor(**kwargs)
        captured.append(agent)
        sdk_headers.append(dict(agent.client.default_headers))
        return agent
    monkeypatch.setattr("run_agent.AIAgent", construct)
    # Stop at the generation boundary, after the production constructor.
    monkeypatch.setattr(real_constructor, "run_conversation", lambda *a, **k: {})
    curator._run_llm_review("synthetic boundary only")
    assert len(captured) == 1
    agent = captured[0]
    assert agent.requested_provider == "custom:beta"
    assert agent.runtime_binding.credential_policy == policy
    assert sdk_headers[0].get("X-Route") == "beta"
    if slot:
        assert agent._credential_pool is None
        assert agent._try_refresh_env_client_credentials() is False


@pytest.mark.parametrize("configured", [False, True])
def test_F5_child_runtime_keeps_own_binding(routes, real_constructor, configured):
    from tools.delegate_tool_config import _resolve_child_runtime, _resolve_delegation_credentials
    from tools.delegate_tool import _credential_overrides

    parent = make_parent(real_constructor)
    child = None
    try:
        creds = _resolve_delegation_credentials({"provider": "custom:alpha"} if configured else {}, parent)
        overrides = _credential_overrides(creds, {})
        overrides.pop("override_request_overrides")
        runtime = _resolve_child_runtime(parent, {}, parent.api_key, model=None, **overrides)
        assert runtime.get("runtime_binding") is not None
        if configured:
            assert runtime["runtime_binding"].requested_provider == "custom:alpha"
            assert runtime["runtime_binding"].credential_policy == "native"
            assert runtime["runtime_binding"] is not parent.runtime_binding
        else:
            assert runtime["runtime_binding"] is parent.runtime_binding
        child = real_constructor(**runtime, quiet_mode=True, skip_memory=True, skip_context_files=True)
        assert child.client.default_headers.get("X-Route") == ("alpha" if configured else "beta")
        if not configured:
            assert child._try_refresh_env_client_credentials() is False
    finally:
        if child is not None:
            child.client.close()
        parent.client.close()


@pytest.mark.parametrize("hint,policy", [({"api_key": "synthetic-config-pin"}, "key_override"),
    ({"base_url": "https://alternate.example/v1"}, "route_override"),
    ({"base_url": "https://alternate.example/v1", "api_key": "synthetic-config-pin"}, "key_override")])
def test_F5_raw_config_intent_survives_bundle(routes, real_constructor, hint, policy):
    from tools.delegate_tool_config import _resolve_delegation_credentials, _resolve_child_runtime
    from tools.delegate_tool import _credential_overrides

    parent = make_parent(real_constructor)
    try:
        creds = _resolve_delegation_credentials({"provider": "custom:alpha", **hint}, parent)
        binding = creds.get("runtime_binding")
        assert binding is not None
        assert binding.requested_provider == "custom:alpha"
        assert binding.credential_policy == policy
        overrides = _credential_overrides(creds, {})
        overrides.pop("override_request_overrides")
        runtime = _resolve_child_runtime(parent, {}, parent.api_key, model=None, **overrides)
        assert runtime["runtime_binding"] is binding
        assert runtime["requested_provider"] == "custom:alpha"
        if "api_key" in hint:
            assert creds["api_key"] == hint["api_key"]
    finally:
        parent.client.close()



def test_F5_contradictory_bundle_rejected(routes, real_constructor):
    from tools.delegate_tool_config import _resolve_child_runtime

    parent = make_parent(real_constructor)
    try:
        with pytest.raises(ValueError, match="binding"):
            _resolve_child_runtime(parent, {}, parent.api_key, model=None, override_provider="custom:alpha",
                override_base_url="https://shared.example/v1", override_api_key="synthetic",
                override_api_mode="chat_completions", override_acp_command=None, override_acp_args=None,
                override_runtime_binding=parent.runtime_binding)
    finally:
        parent.client.close()


@pytest.mark.parametrize("status", [401, 429])
def test_F1_native_parent_fork_shares_pool_and_rotates(routes, real_constructor, status):
    from agent.background_review import build_cache_parity_fork
    import time

    pool = route_fixtures._write_route_pool("beta", "https://shared.example/v1")
    parent = make_parent(real_constructor, "native")
    parent._credential_pool = pool
    fork = None
    try:
        fork, _, _ = build_cache_parity_fork(parent, {}, max_iterations=1)
        assert fork._credential_pool is pool
        before = fork.api_key
        assert fork._recover_with_credential_pool(status_code=status, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600})[0]
        assert fork.api_key != before
        assert fork.client.default_headers.get("X-Route") == "beta"
    finally:
        if fork is not None:
            fork.client.close()
        parent.client.close()


def test_F5_production_child_builder_passes_binding(routes, real_constructor, monkeypatch):
    from tools import delegate_tool

    parent = make_parent(real_constructor)
    seen = []
    class BoundaryReached(Exception):
        pass
    def construct(**kwargs):
        child = real_constructor(**kwargs)
        seen.append((child.runtime_binding, child.requested_provider,
                     child.client.default_headers.get("X-Route")))
        child.client.close()
        raise BoundaryReached()
    monkeypatch.setattr("run_agent.AIAgent", construct)
    monkeypatch.setattr(delegate_tool, "_open_child_session_db", lambda *a: None)
    try:
        # The production builder may translate boundary exceptions into a failed result.
        try:
            delegate_tool._build_child_agent(0, "boundary only", None, [], None, 1, 1, parent)
        except BoundaryReached:
            pass
        assert seen == [(parent.runtime_binding, "custom:beta", "beta")]
    finally:
        parent.client.close()
