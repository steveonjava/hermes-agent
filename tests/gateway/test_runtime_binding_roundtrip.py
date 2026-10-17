"""Nonsecret runtime authority at gateway persistence and reconstruction boundaries."""
import json
from types import SimpleNamespace

import pytest

from gateway.session import sanitize_model_override
from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state


def test_session_codec_roundtrip():
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    record = sanitize_model_override({"model": "m", "provider": "custom:beta",
                                     "runtime_binding": binding, "api_key": "synthetic-private"})
    assert record.get("runtime_binding") == runtime_binding_state(binding)
    assert "api_key" not in record
    assert "synthetic-private" not in json.dumps(record)
    assert sanitize_model_override(json.loads(json.dumps(record))) == record


@pytest.mark.parametrize("bad", [None, {}, {"version": 2}, "invalid"])
def test_session_codec_rejects_present_invalid(bad):
    with pytest.raises(ValueError, match="runtime binding"):
        sanitize_model_override({"model": "m", "runtime_binding": bad})


def test_session_codec_legacy_control():
    assert sanitize_model_override({"model": "m", "api_key": "private"}) == {"model": "m"}


@pytest.mark.parametrize("policy", ["key_override", "route_override", "named_nonpool"])
def test_detached_override_never_discovers_pool(monkeypatch, policy):
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    binding = RuntimeBinding("custom", "custom:beta", policy)
    override = {"model": "m", "provider": "custom:beta", "api_key": "synthetic-pin",
                "runtime_binding": runtime_binding_state(binding)}
    runner = SimpleNamespace(_session_model_override=lambda key: override)
    monkeypatch.setattr("gateway.run._credential_pool_for_provider",
                        lambda p: pytest.fail("pool discovery forbidden"))
    _, runtime = GatewayAgentCacheMixin._apply_session_model_override(
        runner, "s", "default", {"api_key": "default", "credential_pool": object()})
    assert runtime["runtime_binding"] == binding
    assert runtime["credential_pool"] is None
    assert runtime["requested_provider"] == "custom:beta"
    assert runtime["api_key"] == "synthetic-pin"


def test_missing_pin_fails_before_default_merge(monkeypatch):
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    override = {"model": "m", "provider": "custom:beta",
                "runtime_binding": runtime_binding_state(RuntimeBinding("custom", "custom:beta", "key_override"))}
    runner = SimpleNamespace(_session_model_override=lambda key: override)
    defaults = {"api_key": "synthetic-default"}
    with pytest.raises(ValueError, match="[Rr]e-supply"):
        GatewayAgentCacheMixin._apply_session_model_override(runner, "s", "default", defaults)
    assert defaults == {"api_key": "synthetic-default"}


def test_rehydrate_missing_pin_never_resolves(monkeypatch):
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    override = {"model": "m", "provider": "custom:beta",
                "runtime_binding": runtime_binding_state(RuntimeBinding("custom", "custom:beta", "key_override"))}
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                        lambda *a, **k: pytest.fail("must not resolve missing pin"))
    runner = SimpleNamespace(_session_model_override=lambda key: None,
                             session_store=SimpleNamespace(get_model_override=lambda key: override))
    with pytest.raises(ValueError, match="[Rr]e-supply"):
        GatewayAgentCacheMixin._rehydrate_session_model_override(runner, "s")


def _result(binding):
    return SimpleNamespace(new_model="m", target_provider="custom", api_key="synthetic-selected",
                           base_url="https://shared.test/v1", api_mode="chat_completions",
                           runtime_binding=binding, provider_capabilities={}, runtime_capabilities={})


def test_gateway_cached_switch_carries_selected_identity():
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    agent = SimpleNamespace(switch_model=lambda **kw: calls.append(kw))
    runner = SimpleNamespace(_cached_agent_for=lambda key: agent)
    ctx = SimpleNamespace(session_key="s", current_model="old")
    assert GatewayModelCommandsMixin._switch_cached_agent_model(runner, _result(binding), ctx, False) is None
    assert calls[0]["requested_provider"] == "custom:beta"
    assert calls[0]["runtime_binding"] == binding


def test_cli_stage_and_restore_carry_binding(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    import hermes_cli.cli_model_switch_mixin as mod
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    cli = SimpleNamespace(model="old", provider="custom", requested_provider="custom:alpha",
                          agent=SimpleNamespace(switch_model=lambda **kw: calls.append(kw)))
    monkeypatch.setattr(mod, "_resolve_cli_reasoning", lambda cli: None)
    assert CLIModelSwitchMixin._stage_and_swap_model(cli, _result(binding), "old")
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["requested_provider"] == "custom:beta"
    snapshot = CLIModelSwitchMixin._snapshot_model_runtime(cli)
    cli.runtime_binding = None
    CLIModelSwitchMixin._restore_model_runtime_snapshot(cli, snapshot)
    assert cli.runtime_binding == binding
    assert calls[-1]["runtime_binding"] == binding


def test_gateway_constructor_projection():
    from gateway.run import _runtime_agent_kwargs
    binding = RuntimeBinding("custom", "custom:beta")
    assert _runtime_agent_kwargs({"runtime_binding": binding})["runtime_binding"] == binding


def test_cli_new_propagates_selector_binding(monkeypatch):
    from hermes_cli.cli_session_mixin import _reset_model_to_config_default
    import cli
    binding = RuntimeBinding("custom", "custom:beta")
    result = _result(binding)
    result.success = True
    monkeypatch.setattr(cli, "CLI_CONFIG", {"model": {"default": "m"}})
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    calls = []
    cli_obj = SimpleNamespace(model="old", provider="custom", api_key="old", base_url="old",
                              agent=SimpleNamespace(switch_model=lambda **kw: calls.append(kw)))
    _reset_model_to_config_default(cli_obj, silent=True)
    assert calls[0]["runtime_binding"] == binding
    assert cli_obj.runtime_binding == binding
    assert cli_obj.requested_provider == "custom:beta"


def test_cli_resume_missing_pin_before_mutation():
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    binding = runtime_binding_state(RuntimeBinding("custom", "custom:beta", "key_override"))
    cli = SimpleNamespace(model="old", provider="custom", agent=None, _console_print=lambda msg: None)
    with pytest.raises(ValueError, match="[Rr]e-supply"):
        CLIModelSwitchMixin._restore_session_model(cli, {"model": "m", "model_config": {
            "provider": "custom:beta", "runtime_binding": binding}})
    assert cli.model == "old"


def test_cli_current_runtime_carries_binding():
    from hermes_cli.cli_agent_setup_mixin import _current_runtime
    binding = RuntimeBinding("custom", "custom:beta")
    cli = SimpleNamespace(api_key="selected", base_url="url", provider="custom", api_mode="chat_completions",
                          acp_command=None, acp_args=[], runtime_binding=binding)
    assert _current_runtime(cli)["runtime_binding"] == binding


def test_cli_runtime_refresh_passes_existing_binding(monkeypatch):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    calls = []
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    def resolve(**kwargs):
        calls.append(kwargs)
        return {"api_key": "pin", "base_url": "https://shared.test"}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    cli = SimpleNamespace(requested_provider="custom:beta", _explicit_api_key="pin", _explicit_base_url="url",
                          runtime_binding=binding)
    assert CLIAgentSetupMixin._probe_runtime_credentials(cli)[0]
    assert calls[0]["runtime_binding"] == binding


def test_actual_session_entry_codec_save_load(tmp_path):
    from datetime import datetime
    from gateway.session import SessionEntry, SessionStore
    from gateway.config import GatewayConfig
    binding = RuntimeBinding("custom", "custom:beta", "native")
    store = SessionStore(tmp_path, GatewayConfig())
    entry = SessionEntry(session_key="s", session_id="sid", created_at=datetime.now(), updated_at=datetime.now())
    store._entries["s"] = entry
    store.set_model_override("s", {"model": "m", "provider": "custom:beta", "runtime_binding": binding,
                                    "api_key": "synthetic-private"})
    value = store.get_model_override("s")
    assert value["runtime_binding"] == runtime_binding_state(binding)
    saved = entry.to_dict()
    assert "synthetic-private" not in json.dumps(saved)
    loaded = SessionEntry.from_dict(json.loads(json.dumps(saved)))
    assert loaded.model_override == value
    old = dict(saved)
    old["model_override"] = {"model": "legacy"}
    assert SessionEntry.from_dict(old).model_override == {"model": "legacy"}
    with pytest.raises(ValueError):
        store.set_model_override("s", {"model": "bad", "runtime_binding": {"version": 9}})
    assert store.get_model_override("s") == value


@pytest.mark.asyncio
@pytest.mark.parametrize("one_turn", [False, True])
async def test_gateway_committed_override_binding(one_turn):
    from unittest.mock import AsyncMock
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    result = _result(binding)
    result.request_overrides = {}
    result.provider_label = "Beta"
    calls = []
    runner = SimpleNamespace(_session_db=None, _pending_model_notes={}, _session_model_overrides={},
                             _claim_one_turn_restore=lambda *a: None, _evict_cached_agent=lambda k: calls.append(k),
                             async_session_store=SimpleNamespace(set_model_override=AsyncMock()))
    ctx = SimpleNamespace(session_key="s", current_model="old", persist_global=False, restore_snapshot={})
    await GatewayModelCommandsMixin._record_model_switch(runner, result, ctx, source=None, one_turn=one_turn, picker=False)
    override = runner._session_model_overrides["s"]
    assert override["runtime_binding"] == runtime_binding_state(binding)
    assert override["requested_provider"] == "custom:beta"
    assert calls == ["s"]
    assert runner.async_session_store.set_model_override.await_count == (0 if one_turn else 1)


def test_cli_persists_binding_in_existing_route():
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    from unittest.mock import MagicMock
    binding = RuntimeBinding("custom", "custom:beta")
    db = MagicMock()
    cli = SimpleNamespace(_session_db=db, session_id="s")
    CLIModelSwitchMixin._persist_model_switch_to_session(cli, _result(binding))
    cfg = db.patch_session_model_config.call_args.args[1]
    assert cfg["gateway_runtime"]["runtime_binding"] == runtime_binding_state(binding)
    assert cfg["provider"] == "custom:beta"


def test_cli_resume_passes_binding_to_live_agent(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    import hermes_cli.cli_model_switch_mixin as mod
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    resolved_calls = []
    def resolve(**kw):
        resolved_calls.append(kw)
        return {"api_key": "selected", "base_url": "https://shared.test", "runtime_binding": binding}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr(mod, "_resolve_cli_reasoning", lambda cli: None)
    cli = SimpleNamespace(model="old", provider="custom:alpha", api_key="old", base_url="old", api_mode="",
                          agent=SimpleNamespace(switch_model=lambda **kw: calls.append(kw)), _console_print=lambda s: None)
    CLIModelSwitchMixin._restore_session_model(cli, {"model": "m", "model_config": {
        "provider": "custom:beta", "runtime_binding": runtime_binding_state(binding)}})
    assert resolved_calls[0]["runtime_binding"] == binding
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["requested_provider"] == "custom:beta"


def test_cli_constructor_receives_binding(monkeypatch):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from unittest.mock import MagicMock
    import cli as cli_module
    binding = RuntimeBinding("custom", "custom:beta")
    cli = MagicMock()
    cli.agent = None
    cli._session_db = object()
    cli._resumed = False
    cli._single_query_mode = True
    cli.runtime_binding = binding
    cli._ensure_runtime_credentials.return_value = True
    monkeypatch.setattr(cli_module, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **kw: None)
    factory = MagicMock(side_effect=RuntimeError("stop after constructor capture"))
    monkeypatch.setattr("run_agent.AIAgent", factory)
    CLIAgentSetupMixin._init_agent(cli, runtime_override={"runtime_binding": binding})
    assert factory.call_args.kwargs["runtime_binding"] == binding


def test_gateway_rehydrate_resolver_boundary_passes_policy(monkeypatch):
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    def resolve(**kw):
        calls.append(kw)
        return {"provider": "custom", "runtime_binding": binding}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    runtime = _resolve_runtime_agent_kwargs_for_provider("custom:beta", "m", runtime_binding=binding,
                                                        base_url="https://alternate.test")
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["explicit_base_url"] == "https://alternate.test"
    assert runtime["runtime_binding"] == binding


def test_cli_successful_resolution_captures_binding(monkeypatch):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from unittest.mock import MagicMock
    binding = RuntimeBinding("custom", "custom:beta")
    cli = MagicMock()
    cli.runtime_binding = None
    cli._primary_resolution_error = None
    cli.model = "m"
    cli.requested_provider = "custom:beta"
    cli.api_mode = "chat_completions"
    cli.api_key = "old"
    cli.base_url = "old"
    cli._explicit_api_key = None
    cli._explicit_base_url = None
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "api_key": "selected", "base_url": "https://shared.test",
        "runtime_binding": binding, "requested_provider": "custom:beta"})
    CLIAgentSetupMixin._ensure_runtime_credentials(cli)
    assert cli.runtime_binding == binding


def test_cli_failed_swap_rolls_back_binding(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    import hermes_cli.cli_model_switch_mixin as mod
    original = RuntimeBinding("custom", "custom:alpha")
    def fail(**kw):
        raise RuntimeError("build failed")
    cli = SimpleNamespace(model="old", provider="custom", runtime_binding=original,
                          requested_provider="custom:alpha", agent=SimpleNamespace(switch_model=fail))
    monkeypatch.setattr(mod, "_resolve_cli_reasoning", lambda c: None)
    assert not CLIModelSwitchMixin._stage_and_swap_model(cli, _result(RuntimeBinding("custom", "custom:beta")), "old")
    assert cli.runtime_binding == original
    assert cli.requested_provider == "custom:alpha"


def test_stored_route_binding_beats_url_recovery(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import stored_session_route
    binding = RuntimeBinding("custom", "custom:beta")
    monkeypatch.setattr("hermes_cli.runtime_provider.canonical_custom_identity", lambda **kw: "custom:alpha")
    route = stored_session_route({"model": "m", "model_config": {"provider": "custom", "runtime_binding": runtime_binding_state(binding)}},
                                current_model="old", current_provider="custom:alpha")
    assert route[1] == "custom:beta"


def test_cli_pinned_resolution_failure_never_uses_fallback(monkeypatch):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from unittest.mock import MagicMock
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    cli = MagicMock()
    cli.runtime_binding = binding
    cli.model = "m"
    cli._resolve_fallback_runtime.side_effect = AssertionError("fallback forbidden")
    def fail(**kw):
        raise ValueError("credential unavailable")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fail)
    assert not CLIAgentSetupMixin._ensure_runtime_credentials(cli)
    cli._resolve_fallback_runtime.assert_not_called()


def test_cli_legacy_persistence_clears_previous_binding():
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    from unittest.mock import MagicMock
    db = MagicMock()
    cli = SimpleNamespace(_session_db=db, session_id="s")
    CLIModelSwitchMixin._persist_model_switch_to_session(cli, _result(None))
    cfg = db.patch_session_model_config.call_args.args[1]
    assert "runtime_binding" not in cfg["gateway_runtime"]
    assert "runtime_binding" in cfg and cfg["runtime_binding"] is None


def test_detached_binding_cannot_reintroduce_override_pool():
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    override = {"provider": "custom:beta", "api_key": "selected", "credential_pool": object(),
                "requested_provider": "custom:beta", "runtime_binding": runtime_binding_state(binding)}
    runner = SimpleNamespace(_session_model_override=lambda key: override)
    _, runtime = GatewayAgentCacheMixin._apply_session_model_override(runner, "s", "m", {})
    assert runtime["credential_pool"] is None


def test_cli_shallow_legacy_snapshot_forwards_requested_identity():
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    calls = []
    cli = SimpleNamespace(agent=SimpleNamespace(switch_model=lambda **kw: calls.append(kw)))
    CLIModelSwitchMixin._restore_model_runtime_snapshot(cli, {"model": "m", "provider": "custom",
                                                            "requested_provider": "custom:beta"})
    assert calls[0]["requested_provider"] == "custom:beta"


def test_cli_same_route_resume_still_restores_binding(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    import hermes_cli.cli_model_switch_mixin as mod
    binding = RuntimeBinding("custom", "custom:beta", "named_nonpool")
    calls = []
    cli = SimpleNamespace(model="m", provider="custom:beta", api_key="selected", base_url="url", api_mode="",
                          runtime_binding=None, agent=SimpleNamespace(switch_model=lambda **kw: calls.append(kw)),
                          _console_print=lambda s: None)
    monkeypatch.setattr(mod, "_resolve_cli_reasoning", lambda c: None)
    CLIModelSwitchMixin._restore_session_model(cli, {"model": "m", "model_config": {
        "provider": "custom:beta", "runtime_binding": runtime_binding_state(binding)}})
    assert cli.runtime_binding == binding
    assert calls[0]["runtime_binding"] == binding


@pytest.mark.asyncio
async def test_cached_commit_evict_to_constructor_boundary(monkeypatch):
    from unittest.mock import AsyncMock
    from gateway.run import _runtime_agent_kwargs
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    result = _result(binding)
    result.request_overrides = {}
    result.provider_label = "Beta"
    live_calls = []
    agent = SimpleNamespace(switch_model=lambda **kw: live_calls.append(kw))
    overrides = {}
    runner = SimpleNamespace(_cached_agent_for=lambda k: agent, _session_db=None, _pending_model_notes={},
        _session_model_overrides=overrides, _session_model_override=lambda k: overrides.get(k),
        _evict_cached_agent=lambda k: None, async_session_store=SimpleNamespace(set_model_override=AsyncMock()))
    ctx = SimpleNamespace(session_key="s", current_model="old", persist_global=False)
    assert GatewayModelCommandsMixin._switch_cached_agent_model(runner, result, ctx, False) is None
    await GatewayModelCommandsMixin._record_model_switch(runner, result, ctx, source=None, one_turn=False, picker=False)
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda p: pytest.fail("pool forbidden"))
    _, runtime = GatewayAgentCacheMixin._apply_session_model_override(runner, "s", "old", {})
    constructor_kwargs = _runtime_agent_kwargs(runtime)
    assert constructor_kwargs["runtime_binding"] == live_calls[0]["runtime_binding"] == binding
    assert constructor_kwargs["requested_provider"] == "custom:beta"
    assert constructor_kwargs["credential_pool"] is None


@pytest.mark.parametrize("policy", ["key_override", "route_override", "named_nonpool", "native", None])
def test_actual_turn_fast_path_retains_authority(monkeypatch, policy):
    from gateway.run_turn import GatewayTurnMixin
    binding = RuntimeBinding("custom", "custom:beta", policy) if policy else None
    pool = object()
    override = {"model": "m", "provider": "custom:beta", "api_key": "synthetic-selected",
                "base_url": "https://alternate.test", "credential_pool": pool}
    if binding:
        override["runtime_binding"] = runtime_binding_state(binding)
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=override))
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *a: "s",
        _rehydrate_session_model_override=lambda key: None, _peek_session_state=lambda key: state)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: "default")
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda p: pytest.fail("unexpected discovery"))
    model, runtime = GatewayTurnMixin._resolve_session_agent_runtime(runner, session_key="s")
    assert model == "m"
    assert runtime.get("runtime_binding") == binding
    assert runtime["credential_pool"] is (pool if policy in (None, "native") else None)
    assert override["credential_pool"] is pool
    route = GatewayTurnMixin._resolve_turn_agent_config(runner, "hello", model, runtime)
    assert route["runtime"].get("runtime_binding") == binding


@pytest.mark.parametrize("bad", [None, {}, {"version": 99}])
def test_actual_turn_rejects_present_invalid_before_resolution(monkeypatch, bad):
    from gateway.run_turn import GatewayTurnMixin
    state = SimpleNamespace(conversation=SimpleNamespace(model_override={
        "model": "m", "provider": "custom:beta", "api_key": "synthetic", "runtime_binding": bad}))
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *a: "s",
        _rehydrate_session_model_override=lambda key: None, _peek_session_state=lambda key: state)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: "default")
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda p: None)
    with pytest.raises(ValueError, match="runtime binding"):
        GatewayTurnMixin._resolve_session_agent_runtime(runner, session_key="s")


@pytest.mark.parametrize("policy", ["key_override", "route_override", "native", None])
def test_actual_turn_credentialless_policy_boundary(monkeypatch, policy):
    from gateway.run_turn import GatewayTurnMixin
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    binding = RuntimeBinding("custom", "custom:beta", policy) if policy else None
    override = {"model": "m", "provider": "custom:beta", "base_url": "https://alternate.test"}
    if binding:
        override["runtime_binding"] = runtime_binding_state(binding)
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=override))
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *a: "s",
        _rehydrate_session_model_override=lambda key: None, _peek_session_state=lambda key: state,
        _session_state=lambda key: state, _session_model_override=lambda key: override)
    runner._apply_session_model_override = lambda *a: GatewayAgentCacheMixin._apply_session_model_override(runner, *a)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: "default")
    calls = []
    def resolve(provider, **kw):
        calls.append(kw)
        if policy == "key_override":
            pytest.fail("missing pin must not resolve")
        if policy == "route_override":
            assert kw["runtime_binding"] == binding
            assert kw["base_url"] == override["base_url"]
        return {"provider": "custom", "api_key": "synthetic-selected"}
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", resolve)
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda p: None)
    if policy == "key_override":
        with pytest.raises(ValueError, match="[Rr]e-supply"):
            GatewayTurnMixin._resolve_session_agent_runtime(runner, session_key="s")
        assert not calls
    else:
        model, runtime = GatewayTurnMixin._resolve_session_agent_runtime(runner, session_key="s")
        assert model == "m"
        assert runtime.get("runtime_binding") == binding


def test_actual_outer_gateway_loader_legacy_control():
    from gateway.session import SessionStore
    entry = {"session_key": "s", "session_id": "sid", "created_at": "2026-01-01T00:00:00",
             "updated_at": "2026-01-01T00:00:00", "model_override": {"model": "m", "provider": "openrouter"}}
    restored = SessionStore._routing_entry_from_json("s", json.dumps(entry))
    assert restored.model_override == entry["model_override"]


def test_actual_gateway_provider_helper_handoff(monkeypatch):
    from gateway.run_turn import GatewayTurnMixin
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    def resolve(**kw):
        calls.append(kw)
        return {"provider": "custom", "requested_provider": "custom:beta", "runtime_binding": binding,
                "api_key": "synthetic-selected", "base_url": kw["explicit_base_url"]}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    runtime = _resolve_runtime_agent_kwargs_for_provider("custom:beta", "m", binding, "https://alternate.test")
    route = GatewayTurnMixin._resolve_turn_agent_config(SimpleNamespace(), "hello", "m", runtime)
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["explicit_base_url"] == "https://alternate.test"
    assert route["runtime"]["runtime_binding"] == binding
    assert route["runtime"]["requested_provider"] == "custom:beta"


@pytest.mark.parametrize("policy", [None, "native", "route_override"])
def test_actual_gateway_unavailable_override_native_control(monkeypatch, policy):
    from gateway.run_turn import GatewayTurnMixin
    binding = RuntimeBinding("custom", "custom:beta", policy) if policy else None
    override = {"model": "m", "provider": "custom:beta"}
    if binding:
        override["runtime_binding"] = runtime_binding_state(binding)
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=override))
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *a: "s",
        _rehydrate_session_model_override=lambda key: None, _peek_session_state=lambda key: state,
        _session_state=lambda key: state)
    def unavailable(*a, **kw):
        raise RuntimeError("selected route unavailable")
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda cfg: "default")
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", unavailable)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "openrouter", "api_key": "synthetic-default"})
    if policy == "route_override":
        with pytest.raises(RuntimeError, match="selected route unavailable"):
            GatewayTurnMixin._resolve_session_agent_runtime(runner, session_key="s")
    else:
        model, runtime = GatewayTurnMixin._resolve_session_agent_runtime(runner, session_key="s")
        assert model == "default"
        assert runtime["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_actual_hygiene_constructor_receives_resolved_authority(monkeypatch):
    from gateway.run_turn import GatewayTurnMixin
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    async def row(sid):
        return {}
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: calls.append(kw) or SimpleNamespace(**kw))
    monkeypatch.setattr("gateway.run._seed_hygiene_system_prompt", lambda *a: None)
    monkeypatch.setattr("hermes_cli.config.load_config", dict)
    runner = SimpleNamespace(_session_db=SimpleNamespace(get_session=row))
    await GatewayTurnMixin._hmwa_hygiene_build_agent(runner, "m", {
        "runtime_binding": binding, "requested_provider": "custom:beta", "api_key": "synthetic"},
        SimpleNamespace(session_id="s"))
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["requested_provider"] == "custom:beta"


def test_actual_gateway_route_signature_distinguishes_policy():
    from gateway.run_turn import GatewayTurnMixin
    native = {"runtime_binding": RuntimeBinding("custom", "custom:beta", "native")}
    detached = {"runtime_binding": RuntimeBinding("custom", "custom:beta", "route_override")}
    first = GatewayTurnMixin._resolve_turn_agent_config(SimpleNamespace(), "hi", "m", native)
    second = GatewayTurnMixin._resolve_turn_agent_config(SimpleNamespace(), "hi", "m", detached)
    assert first["signature"] != second["signature"]
