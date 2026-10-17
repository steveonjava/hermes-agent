"""Offline real-SDK cross-seam acceptance for a cached gateway route rebuild."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.config import atomic_config_replace
from hermes_cli.runtime_binding import parse_runtime_binding, runtime_binding_state
from tests.agent import test_retained_named_routes as route_fixtures

offline_context_metadata = route_fixtures.offline_context_metadata


def _real_runtime_agent(runtime, **kwargs):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=128_000),
    ):
        return AIAgent(
            model="primary-model", provider=runtime["provider"],
            requested_provider=runtime.get("requested_provider"),
            api_key=runtime["api_key"], base_url=runtime["base_url"], api_mode=runtime["api_mode"],
            credential_pool=runtime.get("credential_pool"),
            runtime_binding=runtime.get("runtime_binding"),
            capabilities=runtime.get("capabilities"), quiet_mode=True,
            skip_context_files=True, skip_memory=True, **kwargs,
        )


def test_gateway_key_pin_survives_live_commit_eviction_and_real_sdk_rebuild(tmp_path, monkeypatch):
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from gateway.session import sanitize_model_override
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    from hermes_cli.model_switch import ModelSwitchResult
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "api_key": f"synthetic-configured-{name}",
               "transport": "chat_completions", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    for name in ("alpha", "beta"):
        route_fixtures._write_route_pool(name, endpoint)
    pinned_key = "synthetic-operator-selected-key"
    beta = resolve_runtime_provider(requested="custom:beta", explicit_api_key=pinned_key,
                                    target_model="next-model")
    assert beta.get("runtime_binding") is not None, "Native resolver discarded credential intent"
    binding = parse_runtime_binding(beta["runtime_binding"])
    assert binding.credential_policy == "key_override"
    assert binding.requested_provider == "custom:beta"
    pin_selected = beta["api_key"] == pinned_key
    assert pin_selected, "The true per-call key must win before any configured pool selection"
    alpha = resolve_runtime_provider(requested="custom:alpha", target_model="primary-model")
    agent = _real_runtime_agent(alpha)
    rebuilt = None
    saved_nonsecret = {}
    evicted = []

    class CodecStoreBoundary:
        async def set_model_override(self, key, value):
            saved_nonsecret[key] = sanitize_model_override(value)

    # This is the real commit/rebuild/codec path, not a live SessionStore durability test.
    runner = SimpleNamespace(
        _cached_agent_for=lambda key: agent,
        _session_model_overrides={}, _session_db=None,
        async_session_store=CodecStoreBoundary(),
        _evict_cached_agent=lambda key: evicted.append(key),
    )
    ctx = SimpleNamespace(session_key="synthetic-session", current_model="primary-model", persist_global=False)
    result = ModelSwitchResult(
        success=True, new_model="next-model", target_provider=beta["provider"],
        api_key=beta["api_key"], base_url=beta["base_url"], api_mode=beta["api_mode"],
        provider_capabilities=beta.get("capabilities") or {}, runtime_capabilities={},
        runtime_binding=binding,
    )
    try:
        error = GatewayModelCommandsMixin._switch_cached_agent_model(runner, result, ctx, False)
        assert error is None
        selected = agent.requested_provider == "custom:beta" and agent.client.default_headers.get("X-Route") == "beta"
        assert selected, "Gateway's actual live switch must carry the destination identity"
        assert agent._credential_pool is None
        unchanged = agent.client.api_key == pinned_key
        assert unchanged
        assert not agent._recover_with_credential_pool(status_code=429, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600})[0]
        asyncio.run(GatewayModelCommandsMixin._record_model_switch(
            runner, result, ctx, source=None, one_turn=False, picker=False))
        assert evicted == [ctx.session_key]
        stored = saved_nonsecret[ctx.session_key]
        assert stored["runtime_binding"] == runtime_binding_state(binding)
        no_secret = pinned_key not in json.dumps(stored)
        assert no_secret, "Persist only validated binding metadata, never the pin"
        cache = SimpleNamespace(_session_model_override=lambda key: runner._session_model_overrides[key])
        _, replay = GatewayAgentCacheMixin._apply_session_model_override(
            cache, ctx.session_key, "primary-model", dict(alpha))
        assert replay["credential_pool"] is None
        assert parse_runtime_binding(replay["runtime_binding"]) == binding
        rebuilt = _real_runtime_agent(replay)
        still_selected = rebuilt.requested_provider == "custom:beta" and rebuilt.client.default_headers.get("X-Route") == "beta"
        still_pinned = rebuilt.client.api_key == pinned_key
        assert still_selected and still_pinned
        assert rebuilt._credential_pool is None
        assert not rebuilt._recover_with_credential_pool(status_code=401, has_retried_429=True)[0]
    finally:
        if rebuilt is not None:
            rebuilt.client.close()
        agent.client.close()


def _routing_row(binding_marker=False, binding=None):
    entry = {"session_key": "s", "session_id": "sid",
             "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00",
             "model_override": {"model": "m", "provider": "openrouter"}}
    if binding_marker:
        entry["model_override"]["runtime_binding"] = binding
    return entry


@pytest.mark.parametrize("bad", [None, {}, {"version": 99}])
def test_actual_routing_row_invalid_present_binding_is_not_absence(bad):
    from gateway.session import SessionStore

    with pytest.raises(ValueError, match="runtime binding"):
        SessionStore._routing_entry_from_json("s", json.dumps(_routing_row(True, bad)))


def test_actual_routing_row_legacy_control():
    from gateway.session import SessionStore

    entry = _routing_row()
    restored = SessionStore._routing_entry_from_json("s", json.dumps(entry))
    assert restored.model_override == entry["model_override"]


@pytest.mark.parametrize("malformation", ["json", "session_id"])
def test_unrelated_corrupt_routing_row_keeps_native_skip_behavior(malformation):
    from gateway.session import SessionStore

    entry = _routing_row()
    entry["session_id"] = "../unsafe-session"
    encoded = "{" if malformation == "json" else json.dumps(entry)
    assert SessionStore._routing_entry_from_json("s", encoded) is None


def _routing_store(tmp_path, monkeypatch, row):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path, GatewayConfig())
    store._db = SimpleNamespace(load_gateway_routing_entries=lambda **kwargs: {"s": json.dumps(row)},
                                get_session=lambda sid: None)
    return store


def test_database_outer_load_cannot_import_legacy_after_invalid_binding(tmp_path, monkeypatch):
    store = _routing_store(tmp_path, monkeypatch, _routing_row(True, {"version": 99}))
    legacy = _routing_row()
    legacy["model_override"]["model"] = "unrelated-fallback-model"
    (tmp_path / "sessions.json").write_text(json.dumps({"s": legacy}))
    with pytest.raises(ValueError, match="runtime binding"):
        store._ensure_loaded()
    assert not store._loaded
    assert "s" not in store._entries


def test_database_outer_load_legacy_record_remains_usable(tmp_path, monkeypatch):
    row = _routing_row()
    store = _routing_store(tmp_path, monkeypatch, row)
    store._ensure_loaded()
    assert store._loaded and store._routing_db_loaded
    assert store.get_model_override("s") == row["model_override"]


@pytest.mark.parametrize("bad", [None, {}, {"version": 99}])
def test_legacy_index_invalid_present_binding_fails_closed(tmp_path, monkeypatch, bad):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path, GatewayConfig())
    store._db = None
    (tmp_path / "sessions.json").write_text(json.dumps({"s": _routing_row(True, bad)}))
    with pytest.raises(ValueError, match="runtime binding"):
        store._ensure_loaded()
    assert not store._loaded
    assert "s" not in store._entries


def test_legacy_index_valid_record_and_unrelated_invalid_json_keep_native_behavior(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    row = _routing_row()
    (tmp_path / "sessions.json").write_text(json.dumps({"s": row}))
    store = SessionStore(tmp_path, GatewayConfig())
    store._db = None
    store._ensure_loaded()
    assert store.get_model_override("s") == row["model_override"]
    (tmp_path / "sessions.json").write_text("{")
    corrupted = SessionStore(tmp_path, GatewayConfig())
    corrupted._db = None
    corrupted._ensure_loaded()
    assert corrupted._loaded and not corrupted._entries


def test_recovered_database_index_preserves_strict_binding_error(tmp_path, monkeypatch):
    store = _routing_store(tmp_path, monkeypatch, _routing_row(True, {"version": 99}))
    store._loaded = True
    store._routing_db_loaded = False
    store._routing_fallback_baseline = {}
    with pytest.raises(ValueError, match="runtime binding"):
        store._ensure_loaded()
    assert not store._routing_db_loaded


def test_invalid_binding_payload_is_never_rendered_by_recovery(tmp_path, monkeypatch, caplog, capsys):
    sentinel = "synthetic-private-value-must-not-render"
    bad = {"version": 1, "provider": "custom", "requested_provider": "custom:beta",
           "credential_policy": "native", "api_key": sentinel}
    store = _routing_store(tmp_path, monkeypatch, _routing_row(True, bad))
    with pytest.raises(ValueError, match="runtime binding") as caught:
        store._ensure_loaded()
    out = capsys.readouterr()
    sanitized = sentinel not in (str(caught.value) + caplog.text + out.out + out.err)
    assert sanitized
    assert not store._loaded


@pytest.mark.parametrize("current_provider", ["custom:beta", "custom"])
@pytest.mark.parametrize("callable_pin", [False, True])
def test_cli_selector_receives_known_current_pin(tmp_path, monkeypatch, current_provider, callable_pin):
    from hermes_cli.cli_model_switch_mixin import _switch_model_from
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://named.example/v1"
    providers = {"beta": {"api": endpoint, "api_key": "synthetic-configured-key",
                           "transport": "chat_completions"}}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": providers})
    monkeypatch.setattr("hermes_cli.model_switch._validate_switch", lambda state: None)
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    pin = (lambda: "synthetic-minted-own-token") if callable_pin else "synthetic-existing-explicit-pin"
    cli = SimpleNamespace(provider=current_provider, requested_provider="custom:beta", model="old-model",
                          base_url=endpoint, api_key=pin, capabilities={}, runtime_binding=binding)
    result = _switch_model_from(cli, "new-model", is_global=False, explicit_provider="",
                                user_providers=providers, custom_providers=[])
    assert result.success
    selected_policy = parse_runtime_binding(result.runtime_binding)
    assert selected_policy == binding, "First-party selector discarded known current credential intent"
    same_key = result.api_key == pin
    assert same_key


@pytest.mark.parametrize("surface", ["gateway", "tui", "acp"])
@pytest.mark.parametrize("callable_key", [False, True])
def test_first_party_selector_current_binding_handoffs(tmp_path, monkeypatch, surface, callable_key):
    from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    captured = {}

    def selector(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(success=False, error_message="synthetic-stop-after-capture")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", selector)
    agent = SimpleNamespace(provider="custom", requested_provider="custom:beta", model="old-model",
        api_key="synthetic-existing-pin", base_url="https://named.example/v1", capabilities={},
        runtime_binding=binding)
    if callable_key:
        agent.api_key = lambda: "synthetic-minted-own-token"
    if surface == "gateway":
        from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext
        ctx = _ModelSwitchContext("s", None, None, False)
        ctx.apply_override({"model": agent.model, "provider": agent.provider, "api_key": agent.api_key,
            "base_url": agent.base_url, "runtime_binding": runtime_binding_state(binding)})
        monkeypatch.setattr("gateway.slash_commands_model._model_switch_skew_guard", lambda: None)
        asyncio.run(GatewayModelCommandsMixin._perform_model_switch(
            SimpleNamespace(), ctx, "new-model", "", None))
    elif surface == "tui":
        from tui_gateway.model_switch import _apply_model_switch
        with pytest.raises(ValueError, match="synthetic-stop-after-capture"):
            _apply_model_switch("s", {"agent": agent}, "new-model")
    else:
        from acp_adapter.server import HermesACPAgent
        state = SimpleNamespace(agent=agent, model=agent.model, session_id="s", cwd=str(tmp_path))
        with pytest.raises(Exception, match="synthetic-stop-after-capture"):
            HermesACPAgent._switch_model(SimpleNamespace(), state, "new-model")
    assert parse_runtime_binding(captured.get("runtime_binding")) == binding, "Selector input lost known current authority"
    preserved_key_source = captured["current_api_key"] is agent.api_key
    assert preserved_key_source, "Selector input stringified the authorized key source"


@pytest.mark.parametrize("has_key", [False, True])
def test_tui_selector_without_agent_uses_known_override_not_profile(tmp_path, monkeypatch, has_key):
    from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state
    from tui_gateway.model_switch import _apply_model_switch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    captured = {}
    def selector(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(success=False, error_message="synthetic-stop-after-capture")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", selector)
    monkeypatch.setattr("tui_gateway.model_switch._current_model_runtime",
                        lambda *args: ("openrouter", "profile-model", "https://profile.example/v1", "synthetic-unrelated-profile-key"))
    override = {"model": "old-model", "provider": "custom", "base_url": "https://named.example/v1",
                "runtime_binding": runtime_binding_state(binding)}
    if has_key:
        override["api_key"] = "synthetic-original-pin"
    with pytest.raises(ValueError, match="synthetic-stop-after-capture"):
        _apply_model_switch("s", {"agent": None, "model_override": override}, "new-model")
    assert parse_runtime_binding(captured.get("runtime_binding")) == binding
    actual_key_source = captured["current_api_key"] == override.get("api_key", "")
    assert actual_key_source
    assert captured["current_base_url"] == override["base_url"]


@pytest.mark.parametrize("surface", ["gateway", "tui"])
def test_selector_context_rejects_serialized_null_binding(tmp_path, monkeypatch, surface):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    if surface == "gateway":
        from gateway.slash_commands_model import _ModelSwitchContext
        ctx = _ModelSwitchContext("s", None, None, False)
        with pytest.raises(ValueError, match="runtime binding"):
            ctx.apply_override({"model": "m", "runtime_binding": None})
    else:
        from tui_gateway.model_switch import _apply_model_switch
        monkeypatch.setattr("tui_gateway.model_switch._current_model_runtime", lambda *args: ("openrouter", "m", "", ""))
        def forbidden(**kwargs):
            pytest.fail("Invalid present binding reached native selector")
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", forbidden)
        with pytest.raises(ValueError, match="runtime binding"):
            _apply_model_switch("s", {"agent": None, "model_override": {"model": "m", "runtime_binding": None}}, "new-model")


def test_gateway_context_missing_pin_never_inherits_previous_key():
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state

    ctx = _ModelSwitchContext("s", None, None, False, current_api_key="synthetic-unrelated-previous-key")
    ctx.apply_override({"model": "m", "runtime_binding": runtime_binding_state(
        RuntimeBinding("custom", "custom:beta", "key_override"))})
    assert not ctx.current_api_key


@pytest.mark.parametrize("shared_endpoint", [False, True])
def test_explicit_named_account_change_leaves_previous_pin(tmp_path, monkeypatch, shared_endpoint):
    from hermes_cli.cli_model_switch_mixin import _switch_model_from
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    alpha_url = "https://alpha.example/v1"
    beta_url = alpha_url if shared_endpoint else "https://beta.example/v1"
    beta_key = "synthetic-configured-beta"
    providers = {"alpha": {"api": alpha_url, "api_key": "synthetic-configured-alpha"},
                 "beta": {"api": beta_url, "api_key": beta_key}}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": providers})
    monkeypatch.setattr("hermes_cli.model_switch._validate_switch", lambda state: None)
    cli = SimpleNamespace(provider="custom:alpha", requested_provider="custom:alpha", model="old-model",
        base_url=alpha_url, api_key="synthetic-previous-alpha-pin", capabilities={},
        runtime_binding=RuntimeBinding("custom", "custom:alpha", "key_override"))
    result = _switch_model_from(cli, "new-model", is_global=False, explicit_provider="beta",
                                user_providers=providers, custom_providers=[])
    assert result.success, result.error_message
    binding = parse_runtime_binding(result.runtime_binding)
    assert binding.requested_provider == "custom:beta" and binding.credential_policy == "native"
    beta_selected = result.api_key == beta_key
    assert beta_selected, "Selected account must supply its own credential, never a same-host sibling's pin"

_GATEWAY_RESTART_CONTROL = r'''
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from gateway.config import GatewayConfig
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.session import SessionStore
from hermes_cli.runtime_binding import parse_runtime_binding
from hermes_cli.runtime_provider import resolve_runtime_provider
from tests.gateway.test_runtime_binding_integration import _real_runtime_agent


def no_network(*args, **kwargs):
    raise AssertionError("Restart control must not use network")


with patch("socket.socket.connect", no_network), patch(
    "agent.model_metadata.get_model_context_length", return_value=128_000
):
    home, session_key, policy = sys.argv[1:]
    store = SessionStore(Path(home) / "sessions", GatewayConfig())
    saved = store.get_model_override(session_key)
    assert parse_runtime_binding(saved["runtime_binding"]).credential_policy == policy
    assert "api_key" not in saved
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=None))
    cache = SimpleNamespace(
        session_store=store,
        _session_model_override=lambda key: state.conversation.model_override,
        _session_state=lambda key: state,
    )
    if policy == "key_override":
        try:
            GatewayAgentCacheMixin._rehydrate_session_model_override(cache, session_key)
        except ValueError as exc:
            assert "re-supply" in str(exc).lower()
        else:
            raise AssertionError("Persisted pin borrowed a profile credential")
        assert state.conversation.model_override is None
        outcome = {"pin_refused": True, "record_policy": policy}
    else:
        GatewayAgentCacheMixin._rehydrate_session_model_override(cache, session_key)
        default = resolve_runtime_provider(requested="custom:alpha", target_model="primary-model")
        _, runtime = GatewayAgentCacheMixin._apply_session_model_override(
            cache, session_key, "primary-model", default
        )
        agent = _real_runtime_agent(runtime)
        try:
            assert agent.requested_provider == "custom:beta"
            assert agent.runtime_binding.credential_policy == policy
            assert agent.client.default_headers.get("X-Route") == "beta"
            outcome = {"selected_beta": True, "record_policy": policy}
        finally:
            agent.client.close()
    print(json.dumps(outcome, sort_keys=True))
'''


@pytest.mark.parametrize("policy", ["native", "key_override"])
def test_actual_gateway_database_fresh_process_restores_or_refuses_pin(tmp_path, monkeypatch, policy):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from hermes_cli.runtime_binding import RuntimeBinding
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    from hermes_cli.model_switch import ModelSwitchResult
    import asyncio

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "api_key": "synthetic-configured-" + name,
               "transport": "chat_completions", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = store.get_or_create_session(SessionSource(
        platform=Platform.TELEGRAM, user_id="synthetic-user", chat_id="synthetic-chat", chat_type="dm"
    ))
    binding = RuntimeBinding("custom", "custom:beta", policy)
    class AsyncStoreBoundary:
        async def set_model_override(self, key, override):
            store.set_model_override(key, override)

    runner = SimpleNamespace(
        _session_db=None, _pending_model_notes={}, _session_model_overrides={},
        async_session_store=AsyncStoreBoundary(), _evict_cached_agent=lambda key: None,
    )
    ctx = SimpleNamespace(session_key=entry.session_key, current_model="primary-model", persist_global=False)
    result = ModelSwitchResult(
        success=True, new_model="selected-model", target_provider="custom", api_key="synthetic-exclusive-pin",
        base_url=endpoint, api_mode="chat_completions", runtime_binding=binding,
    )
    asyncio.run(GatewayModelCommandsMixin._record_model_switch(
        runner, result, ctx, source=None, one_turn=False, picker=False,
    ))
    record = store.get_model_override(entry.session_key)
    assert record["runtime_binding"] == runtime_binding_state(binding)
    assert "synthetic-exclusive-pin" not in json.dumps(record)
    for db in list(store._db_handles.values()):
        db.close()
    env = {"HOME": str(tmp_path), "HERMES_HOME": str(tmp_path),
           "PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _GATEWAY_RESTART_CONTROL, str(tmp_path), entry.session_key, policy],
        cwd=Path(sys.modules["run_agent"].__file__).resolve().parent, env=env,
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout.splitlines()[-1])
    assert outcome["record_policy"] == policy
    assert outcome.get("selected_beta" if policy == "native" else "pin_refused") is True
