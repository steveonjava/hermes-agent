"""Runtime authority survives existing desktop persistence and JSON boundaries."""
import json
from types import SimpleNamespace

import pytest

from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state
from tests.agent import test_retained_named_routes as route_fixtures
from tests.gateway.test_runtime_binding_integration import _real_runtime_agent

offline_context_metadata = route_fixtures.offline_context_metadata


def test_shared_url_persists_selected_binding(monkeypatch):
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    monkeypatch.setattr("hermes_cli.runtime_provider.canonical_custom_identity", lambda **kw: "custom:alpha")
    agent = SimpleNamespace(model="m", provider="custom", base_url="https://shared.test/v1",
                            runtime_binding=binding)
    cfg = server._runtime_model_config(agent)
    assert cfg["provider"] == "custom:beta"
    assert cfg["runtime_binding"] == runtime_binding_state(binding)
    row = {"model": "m", "model_config": json.dumps(cfg)}
    monkeypatch.setattr(server, "_is_routable_provider", lambda p: True)
    restored = server._stored_session_runtime_overrides(row)["model_override"]
    assert restored["runtime_binding"] == runtime_binding_state(binding)
    assert restored["provider"] == "custom:beta"


@pytest.mark.parametrize("nested", [False, True])
def test_common_reader_binding(nested):
    from hermes_state import SessionDB
    binding = runtime_binding_state(RuntimeBinding("custom", "custom:beta"))
    route = {"provider": "custom:beta", "runtime_binding": binding}
    cfg = {"gateway_runtime": route} if nested else route
    assert SessionDB.session_gateway_runtime({"model_config": json.dumps(cfg)})["runtime_binding"] == binding


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("bad", [None, {}, {"version": 9}])
def test_common_reader_invalid_binding(nested, bad):
    from hermes_state import SessionDB
    route = {"provider": "custom:beta", "runtime_binding": bad}
    cfg = {"gateway_runtime": route} if nested else route
    with pytest.raises(ValueError, match="runtime binding"):
        SessionDB.session_gateway_runtime({"model_config": json.dumps(cfg)})


def test_tui_shallow_restore_carries_binding():
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    agent = SimpleNamespace(model="m", provider="custom", requested_provider="custom:beta",
                            api_key="selected", base_url="https://shared.test", api_mode="chat_completions",
                            runtime_binding=binding, switch_model=lambda **kw: calls.append(kw))
    snapshot = server._snapshot_agent_model_runtime(agent)
    agent.runtime_binding = None
    server._restore_agent_model_runtime(agent, snapshot)
    assert calls[-1]["requested_provider"] == "custom:beta"
    assert calls[-1]["runtime_binding"] == binding


def test_tui_commit_carries_binding(monkeypatch):
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    agent = SimpleNamespace(switch_model=lambda **kw: calls.append(kw))
    result = SimpleNamespace(new_model="m", target_provider="custom", api_key="selected",
                             base_url="https://shared.test", api_mode="chat_completions", runtime_binding=binding)
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    server._commit_agent_switch("s", {}, agent, result, "old", None)
    assert calls[-1]["runtime_binding"] == binding
    assert calls[-1]["requested_provider"] == "custom:beta"


def test_tui_missing_pin_fails_before_resolution(monkeypatch):
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    monkeypatch.setattr(server, "_resolve_runtime_with_fallback", lambda *a: pytest.fail("must fail before resolution"))
    with pytest.raises(ValueError, match="[Rr]e-supply"):
        server._resolve_agent_model_runtime({"model": "m", "runtime_binding": runtime_binding_state(binding)}, None)


def test_tui_rebuild_passes_binding_to_resolver(monkeypatch):
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    calls = []
    def resolve(kwargs):
        calls.append(kwargs)
        return server._RuntimeFallbackResolution({"provider": "custom", "api_key": "selected"}, None, False)
    monkeypatch.setattr(server, "_resolve_runtime_with_fallback", resolve)
    monkeypatch.setattr(server, "_rederive_per_model_route", lambda *a: None)
    _, runtime = server._resolve_agent_model_runtime({"model": "m", "provider": "custom",
        "runtime_binding": runtime_binding_state(binding), "base_url": "https://shared.test"}, None)
    assert calls[0]["requested"] == "custom:beta"
    assert calls[0]["runtime_binding"] == binding
    assert runtime["runtime_binding"] == binding
    assert runtime["credential_pool"] is None


def test_compute_host_frame_json_binding(monkeypatch):
    import threading
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    session = {"history_lock": threading.Lock(), "model_override": {"model": "m", "runtime_binding": binding}}
    for name in ("_session_cwd", "_session_source", "_session_auth_user_id", "_active_session_lease_vouch"):
        monkeypatch.setattr(server, name, lambda s: None)
    monkeypatch.setattr(server, "_context_cwd_is_launch_artifact", lambda s: False)
    frame = server._compute_host_turn_frame("r", "s", session, "hello")
    decoded = json.loads(json.dumps(frame))
    assert decoded["model_override"]["runtime_binding"] == runtime_binding_state(binding)
    assert session["model_override"]["runtime_binding"] is binding


def test_tui_constructor_receives_binding(monkeypatch):
    from tui_gateway import server
    from unittest.mock import MagicMock
    binding = RuntimeBinding("custom", "custom:beta")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: ("m", {"runtime_binding": binding}))
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_startup_system_prompt", lambda *a: "")
    monkeypatch.setattr(server, "_load_provider_routing", dict)
    monkeypatch.setattr(server, "_agent_cbs", lambda *a: {})
    factory = MagicMock()
    monkeypatch.setattr("run_agent.AIAgent", factory)
    server._make_agent("s", "k")
    assert factory.call_args.kwargs["runtime_binding"] == binding


def test_compute_host_receiver_rejects_invalid_before_build():
    from tui_gateway.compute_host import ComputeHost
    host = object.__new__(ComputeHost)
    host._transport = None
    server = SimpleNamespace(_sessions={}, _make_agent=lambda *a, **k: pytest.fail("construction forbidden"))
    with pytest.raises(ValueError, match="runtime binding"):
        host._ensure_server_session(server, {"sid": "s", "model_override": {"model": "m", "runtime_binding": None}})


def test_tui_once_does_not_persist_temporary_runtime(monkeypatch):
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta")
    agent = SimpleNamespace(switch_model=lambda **kw: None)
    result = SimpleNamespace(new_model="m", target_provider="custom", api_key="selected", base_url="url",
                             api_mode="chat_completions", runtime_binding=binding)
    calls = []
    for name in ("_restart_slash_worker", "_persist_live_session_system_prompt", "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *a: calls.append(True))
    session = {"one_turn_model_restore": {"earliest": True}}
    server._commit_agent_switch("s", session, agent, result, "old", {"later": True})
    assert not calls
    assert session["one_turn_model_restore"] == {"earliest": True}


def test_tui_live_override_contains_json_binding(monkeypatch):
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta")
    result = SimpleNamespace(success=True, new_model="m", target_provider="custom", api_key="selected", base_url="url",
                             api_mode="chat_completions", runtime_binding=binding, warning_message="")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    monkeypatch.setattr(server, "_current_model_runtime", lambda *a: ("custom", "old", "url", "key"))
    monkeypatch.setattr(server, "_switch_request", lambda *a: ("m", None, False, False, None))
    session = {}
    server._apply_model_switch("s", session, "m", confirm_expensive_model=True, count_switch=False)
    assert session["model_override"]["runtime_binding"] == runtime_binding_state(binding)
    json.dumps(session["model_override"])


def test_compute_host_return_binding_replaces_mirror(monkeypatch):
    import threading
    from tui_gateway import server
    binding = runtime_binding_state(RuntimeBinding("custom", "custom:beta", "key_override"))
    monkeypatch.setattr(server, "_compute_host_adopt_frame_meta", lambda *a: None)
    frame = json.loads(json.dumps({"model_override": {"model": "m", "provider": "custom:beta", "runtime_binding": binding}}))
    session = {"history_lock": threading.Lock()}
    server._apply_compute_host_metadata_mirror(session, frame)
    assert session["model_override"]["runtime_binding"] == binding


def test_compute_host_history_meta_serializes_nonsecret_override():
    from tui_gateway.compute_host import _history_meta
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    frame = json.loads(json.dumps(_history_meta({"history": [], "model_override": {
        "model": "m", "runtime_binding": binding, "api_key": "synthetic-private"}})))
    assert frame["model_override"]["runtime_binding"] == runtime_binding_state(binding)
    assert "synthetic-private" not in json.dumps(frame)


def test_tui_pinned_auth_error_never_falls_back(monkeypatch):
    from hermes_cli.auth import AuthError
    from tui_gateway import server
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    calls = []
    def fail(**kw):
        calls.append(kw)
        raise AuthError("unavailable", provider="custom")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fail)
    monkeypatch.setattr(server, "_load_fallback_model", lambda: [{"provider": "openrouter", "model": "other"}])
    with pytest.raises(AuthError):
        server._resolve_agent_model_runtime({"model": "m", "api_key": "pin",
                                             "runtime_binding": runtime_binding_state(binding)}, None)
    assert len(calls) == 1


def test_compute_host_receiver_json_roundtrip_to_constructor(monkeypatch):
    import threading
    from tui_gateway import server
    from tui_gateway.compute_host import ComputeHost, _history_meta
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    session = {"history_lock": threading.Lock(), "model_override": {"model": "m", "provider": "custom:beta",
                                                                              "runtime_binding": binding}}
    for name in ("_session_cwd", "_session_source", "_session_auth_user_id", "_active_session_lease_vouch"):
        monkeypatch.setattr(server, name, lambda s: None)
    monkeypatch.setattr(server, "_context_cwd_is_launch_artifact", lambda s: False)
    frame = json.loads(json.dumps(server._compute_host_turn_frame("r", "s", session, "hello")))
    calls = []
    child_session = {"history": [], "history_lock": threading.Lock()}
    def build(*a, **kw):
        calls.append(kw)
        return SimpleNamespace()
    child = SimpleNamespace(_sessions={}, _make_agent=build, _transfer_db_to_agent=lambda *a: False,
        _init_session=lambda *a, **kw: child._sessions.update(s=child_session))
    host = object.__new__(ComputeHost)
    host._transport = object()
    restored = host._ensure_server_session(child, frame)
    assert calls[0]["model_override"]["runtime_binding"] == runtime_binding_state(binding)
    returned = json.loads(json.dumps(_history_meta(restored)))
    server._apply_compute_host_metadata_mirror(session, returned)
    assert session["model_override"]["runtime_binding"] == runtime_binding_state(binding)


def test_once_postturn_writer_and_return_keep_standing_binding(monkeypatch):
    from tui_gateway import server
    from tui_gateway.compute_host import _history_meta
    standing = RuntimeBinding("custom", "custom:beta", "native")
    temporary = RuntimeBinding("custom", "custom:alpha", "key_override")
    writes = []
    db = SimpleNamespace(get_session=lambda key: {}, update_session_meta=lambda *a: writes.append(a))
    agent = SimpleNamespace(model="standing", provider="custom", requested_provider="custom:beta",
        base_url="https://standing.test", api_key="synthetic-standing", api_mode="chat_completions",
        runtime_binding=standing, _session_db=db)
    def switch(**kw):
        agent.model = kw["new_model"]
        agent.provider = kw["new_provider"]
        for key in ("runtime_binding", "requested_provider", "base_url", "api_key", "api_mode"):
            if key in kw:
                setattr(agent, key, kw[key])
    agent.switch_model = switch
    session = {"agent": agent, "session_key": "s", "model_override": {
        "model": "standing", "provider": "custom:beta", "runtime_binding": runtime_binding_state(standing)}}
    for name in ("_restart_slash_worker", "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    first = server._snapshot_agent_model_runtime(agent)
    result = SimpleNamespace(new_model="temporary", target_provider="custom", api_key="synthetic-temp",
        base_url="https://temporary.test", api_mode="chat_completions", runtime_binding=temporary)
    server._commit_agent_switch("s", session, agent, result, "standing", first)
    server._commit_agent_switch("s", session, agent, result, "temporary", server._snapshot_agent_model_runtime(agent))
    assert session["one_turn_model_restore"] is first
    active_snapshot = session.pop("one_turn_model_restore")
    server._persist_live_session_runtime(session)
    meta = json.loads(writes[-1][1])
    assert meta["model"] == "standing"
    assert meta["runtime_binding"] == runtime_binding_state(standing)
    returned = json.loads(json.dumps(_history_meta(session)))
    assert returned["model_override"]["runtime_binding"] == runtime_binding_state(standing)
    server._restore_agent_model_runtime(agent, active_snapshot)
    server._persist_live_session_runtime(session)
    assert agent.runtime_binding == standing
    assert json.loads(writes[-1][1])["runtime_binding"] == runtime_binding_state(standing)


@pytest.mark.parametrize("bad", [None, {}, {"version": 99}])
def test_actual_tui_outer_stored_route_invalid_never_downgrades(bad):
    from tui_gateway import server
    row = {"model": "m", "model_config": json.dumps({"provider": "custom:beta", "runtime_binding": bad})}
    with pytest.raises(ValueError, match="runtime binding"):
        server._stored_session_runtime_overrides(row)


def test_once_then_standing_commit_clears_writer_snapshot(monkeypatch):
    from tui_gateway import server
    agent = SimpleNamespace(switch_model=lambda **kw: None, _one_turn_runtime_restore={"model": "old"})
    result = SimpleNamespace(new_model="standing", target_provider="custom", api_key="synthetic",
        base_url="https://selected.test", api_mode="chat_completions", runtime_binding=None)
    session = {"one_turn_model_restore": {"model": "old"}}
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    server._commit_agent_switch("s", session, agent, result, "old", None)
    assert getattr(agent, "_one_turn_runtime_restore", None) is None
    assert "one_turn_model_restore" not in session



def _real_pooled_primary(tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_replace
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "api_key": "synthetic-configured-" + name,
               "transport": "chat_completions", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    route_fixtures._write_route_pool("beta", endpoint)
    runtime = resolve_runtime_provider(requested="custom:beta", target_model="primary-model")
    return _real_runtime_agent(runtime), runtime, endpoint


@pytest.mark.parametrize("status", [401, 429])
def test_real_native_pool_recovery_control(tmp_path, monkeypatch, status):
    import time

    agent, _, _ = _real_pooled_primary(tmp_path, monkeypatch)
    try:
        assert agent._credential_pool is not None
        before = agent.api_key
        assert agent._recover_with_credential_pool(
            status_code=status, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600},
        )[0]
        rotated = agent.api_key != before
        assert rotated
        assert agent.client.default_headers.get("X-Route") == "beta"
    finally:
        agent.client.close()


@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.parametrize("primary_origin", ["constructor", "public_switch"])
@pytest.mark.parametrize("repeat_once", [False, True])
def test_real_one_turn_finally_restores_native_pool_recovery(tmp_path, monkeypatch, status, primary_origin, repeat_once):
    import time
    from tui_gateway import server

    agent, runtime, endpoint = _real_pooled_primary(tmp_path, monkeypatch)
    writes = []
    # Capture only the durable-writer boundary; route, SDK, snapshots and recovery stay real.
    agent._session_db = SimpleNamespace(
        get_session=lambda key: {}, update_session_meta=lambda *args: writes.append(args),
        update_session_billing_route=lambda *args, **kwargs: None,
    )
    session = {"agent": agent, "session_key": "synthetic-session", "model_override": {
        "model": agent.model, "provider": "custom:beta",
        "runtime_binding": runtime_binding_state(agent.runtime_binding),
    }}
    for name in ("_restart_slash_worker", "_append_model_switch_marker",
                 "_emit_session_info", "_persist_live_session_system_prompt"):
        monkeypatch.setattr(server, name, lambda *args, **kwargs: None)
    if primary_origin == "public_switch":
        agent.switch_model(new_model="switched-primary", new_provider=runtime["provider"],
            requested_provider=runtime["requested_provider"], runtime_binding=runtime["runtime_binding"],
            api_key=runtime["api_key"], base_url=runtime["base_url"], api_mode=runtime["api_mode"])
        assert agent._primary_runtime["model"] == "switched-primary"
    snapshot = server._snapshot_agent_model_runtime(agent)
    temporary = SimpleNamespace(
        new_model="temporary-model", target_provider="custom", api_key="synthetic-once-pin",
        base_url=endpoint, api_mode="chat_completions",
        runtime_binding=RuntimeBinding("custom", "custom:alpha", "key_override"),
    )
    try:
        assert agent._credential_pool is not None
        server._commit_agent_switch("synthetic-session", session, agent, temporary, agent.model, snapshot)
        assert agent.requested_provider == "custom:alpha"
        assert agent._credential_pool is None
        if repeat_once:
            later = server._snapshot_agent_model_runtime(agent)
            server._commit_agent_switch("synthetic-session", session, agent, temporary, agent.model, later)
            assert session["one_turn_model_restore"] is snapshot
        turn = server._TurnRun(agent, session.pop("one_turn_model_restore"), None, True)
        server._release_turn_scopes("synthetic-session", session, turn)
        assert agent.requested_provider == "custom:beta"
        assert agent.runtime_binding == runtime["runtime_binding"]
        assert agent.client.default_headers.get("X-Route") == "beta"
        assert agent._credential_pool is not None
        assert agent._credential_pool.provider == "beta"
        restored = json.loads(writes[-1][1])
        assert restored["runtime_binding"] == runtime_binding_state(runtime["runtime_binding"])
        assert "primary_had_credential_pool" not in restored
        assert "_primary_had_credential_pool" not in restored
        no_pin = "synthetic-once-pin" not in json.dumps(restored)
        assert no_pin
        before = agent.api_key
        assert agent._recover_with_credential_pool(
            status_code=status, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600},
        )[0]
        rotated = agent.api_key != before
        assert rotated
        assert agent.client.default_headers.get("X-Route") == "beta"
    finally:
        agent.client.close()

@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.parametrize("primary_origin", ["constructor", "public_switch"])
def test_real_cli_preferred_restore_keeps_native_pool_recovery(tmp_path, monkeypatch, status, primary_origin):
    import time
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    from hermes_cli.model_switch import ModelSwitchResult

    agent, runtime, endpoint = _real_pooled_primary(tmp_path, monkeypatch)
    if primary_origin == "public_switch":
        agent.switch_model(new_model="switched-primary", new_provider=runtime["provider"],
            requested_provider=runtime["requested_provider"], runtime_binding=runtime["runtime_binding"],
            api_key=runtime["api_key"], base_url=runtime["base_url"], api_mode=runtime["api_mode"])
        assert agent._primary_runtime["model"] == "switched-primary"
    cli = SimpleNamespace(
        agent=agent, model=agent.model, provider=agent.provider,
        requested_provider=agent.requested_provider, runtime_binding=agent.runtime_binding,
        api_key=agent.api_key, base_url=agent.base_url, api_mode=agent.api_mode,
    )
    monkeypatch.setattr("hermes_cli.cli_model_switch_mixin._resolve_cli_reasoning", lambda obj: None)
    snapshot = CLIModelSwitchMixin._snapshot_model_runtime(cli)
    temporary = ModelSwitchResult(
        success=True, new_model="temporary-model", target_provider="custom",
        api_key="synthetic-once-pin", base_url=endpoint, api_mode="chat_completions",
        runtime_binding=RuntimeBinding("custom", "custom:alpha", "key_override"),
    )
    try:
        assert CLIModelSwitchMixin._stage_and_swap_model(cli, temporary, cli.model)
        assert agent.requested_provider == "custom:alpha"
        assert agent._credential_pool is None
        CLIModelSwitchMixin._restore_model_runtime_snapshot(cli, snapshot)
        assert cli.runtime_binding == runtime["runtime_binding"]
        assert agent.runtime_binding == runtime["runtime_binding"]
        assert agent.requested_provider == "custom:beta"
        assert agent.client.default_headers.get("X-Route") == "beta"
        assert agent._credential_pool is not None
        assert agent._credential_pool.provider == "beta"
        before = agent.api_key
        assert agent._recover_with_credential_pool(
            status_code=status, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600},
        )[0]
        rotated = agent.api_key != before
        assert rotated
        assert agent.client.default_headers.get("X-Route") == "beta"
    finally:
        agent.client.close()


def _begin_real_temporary_surface(agent, standing, endpoint, surface, monkeypatch):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    from hermes_cli.model_switch import ModelSwitchResult
    from tui_gateway import server

    temporary = ModelSwitchResult(
        success=True, new_model="temporary-model", target_provider="custom",
        api_key="synthetic-once-pin", base_url=endpoint, api_mode="chat_completions",
        runtime_binding=RuntimeBinding("custom", "custom:alpha", "key_override"),
    )
    if surface == "cli":
        cli = SimpleNamespace(
            agent=agent, model=agent._primary_runtime["model"], provider=standing["provider"],
            requested_provider=standing.get("requested_provider"), runtime_binding=standing.get("runtime_binding"),
            api_key=standing["api_key"], base_url=standing["base_url"], api_mode=standing["api_mode"],
        )
        monkeypatch.setattr("hermes_cli.cli_model_switch_mixin._resolve_cli_reasoning", lambda obj: None)
        snapshot = CLIModelSwitchMixin._snapshot_model_runtime(cli)
        assert CLIModelSwitchMixin._stage_and_swap_model(cli, temporary, cli.model)
        return lambda: CLIModelSwitchMixin._restore_model_runtime_snapshot(cli, snapshot)
    writes = []
    agent._session_db = SimpleNamespace(
        get_session=lambda key: {}, update_session_meta=lambda *args: writes.append(args),
        update_session_billing_route=lambda *args, **kwargs: None,
    )
    session = {"agent": agent, "session_key": "synthetic-session", "model_override": {
        "model": agent.model, "provider": standing.get("requested_provider") or standing["provider"],
        **({"runtime_binding": runtime_binding_state(standing["runtime_binding"])}
           if standing.get("runtime_binding") is not None else {}),
    }}
    for name in ("_restart_slash_worker", "_append_model_switch_marker",
                 "_emit_session_info", "_persist_live_session_system_prompt"):
        monkeypatch.setattr(server, name, lambda *args, **kwargs: None)
    snapshot = server._snapshot_agent_model_runtime(agent)
    server._commit_agent_switch("synthetic-session", session, agent, temporary, agent.model, snapshot)

    def restore():
        turn = server._TurnRun(agent, session.pop("one_turn_model_restore"), None, True)
        server._release_turn_scopes("synthetic-session", session, turn)
    return restore


@pytest.mark.parametrize("surface", ["tui", "cli"])
@pytest.mark.parametrize("primary_pooled", [False, True])
def test_real_active_fallback_snapshot_keeps_primary_pool_provenance(tmp_path, monkeypatch, surface, primary_pooled):
    import time
    from unittest.mock import patch

    initial, beta, endpoint = _real_pooled_primary(tmp_path, monkeypatch)
    initial.client.close()
    primary = "beta" if primary_pooled else "alpha"
    # Legacy public constructors can intentionally start a selected named runtime without a pool.
    runtime = beta if primary_pooled else {
        "provider": "custom", "requested_provider": "custom:alpha", "api_key": "synthetic-standing-nonpool",
        "base_url": endpoint, "api_mode": "chat_completions"}
    fallback = {"provider": "custom:alpha", "model": "fallback-model", "api_key": "synthetic-fallback-pin"} if primary_pooled else {
        "provider": "custom:beta", "model": "fallback-model"}
    agent = _real_runtime_agent(runtime, fallback_model=fallback)
    try:
        assert (agent._credential_pool is not None) is primary_pooled
        assert agent._try_activate_fallback()
        assert agent._primary_had_credential_pool is primary_pooled
        assert (agent._credential_pool is not None) is not primary_pooled
        if not primary_pooled:
            # Adding a pool later cannot rewrite the known nonpooled departure.
            route_fixtures._write_route_pool("alpha", endpoint)
        restore = _begin_real_temporary_surface(agent, runtime, endpoint, surface, monkeypatch)
        if primary_pooled:
            restore()
            assert agent._credential_pool is not None
            assert agent._credential_pool.provider == primary
            assert agent._recover_with_credential_pool(status_code=429, has_retried_429=True,
                error_context={"reset_at": time.time() + 3600})[0]
        else:
            with patch("agent.credential_pool.load_pool", side_effect=AssertionError("known nonpool primary must not discover")) as load:
                restore()
            load.assert_not_called()
            assert agent._credential_pool is None
        assert agent.requested_provider == "custom:" + primary
        assert agent.client.default_headers.get("X-Route") == primary
    finally:
        agent.client.close()


@pytest.mark.parametrize("surface", ["tui", "cli"])
@pytest.mark.parametrize("policy", ["key_override", "route_override"])
def test_real_one_turn_return_never_reattaches_exclusive_primary(tmp_path, monkeypatch, surface, policy):
    from unittest.mock import patch
    from hermes_cli.runtime_provider import resolve_runtime_provider

    initial, _, endpoint = _real_pooled_primary(tmp_path, monkeypatch)
    initial.client.close()
    explicit = {"explicit_api_key": "synthetic-standing-pin"} if policy == "key_override" else {
        "explicit_base_url": "https://alternate.example/v1"}
    runtime = resolve_runtime_provider(requested="custom:beta", target_model="primary-model", **explicit)
    agent = _real_runtime_agent(runtime)
    try:
        assert agent.runtime_binding.credential_policy == policy
        assert agent._credential_pool is None
        restore = _begin_real_temporary_surface(agent, runtime, endpoint, surface, monkeypatch)
        with patch("agent.credential_pool.load_pool", side_effect=AssertionError("exclusive primary must not discover")) as load:
            restore()
        load.assert_not_called()
        assert agent.runtime_binding == runtime["runtime_binding"]
        assert agent._credential_pool is None
        unchanged = agent.api_key == runtime["api_key"]
        assert unchanged
        assert str(agent.base_url).rstrip("/") == runtime["base_url"].rstrip("/")
        assert agent.client.default_headers.get("X-Route") == "beta"
        assert not agent._recover_with_credential_pool(status_code=401, has_retried_429=True)[0]
        assert not agent._recover_with_credential_pool(status_code=429, has_retried_429=True)[0]
    finally:
        agent.client.close()


@pytest.mark.parametrize("surface", ["tui", "cli"])
@pytest.mark.parametrize("provider", ["openai", "custom"])
def test_real_legacy_none_pool_one_turn_keeps_native_lazy_semantics(tmp_path, monkeypatch, surface, provider):
    from unittest.mock import patch
    from hermes_cli.runtime_binding import allows_native_pool

    initial, _, endpoint = _real_pooled_primary(tmp_path, monkeypatch)
    initial.client.close()
    runtime = {"provider": provider, "requested_provider": provider, "api_key": "synthetic-native-key",
               "base_url": "https://plain.example/v1", "api_mode": "chat_completions"}
    agent = _real_runtime_agent(runtime)
    try:
        assert agent._credential_pool is None
        restore = _begin_real_temporary_surface(agent, runtime, endpoint, surface, monkeypatch)
        with patch("agent.credential_pool.load_pool", return_value=None) as load:
            restore()
        load.assert_not_called()
        assert agent.provider == provider
        assert agent._credential_pool is None
        assert allows_native_pool(agent.runtime_binding)
        unchanged = agent.api_key == runtime["api_key"]
        assert unchanged
    finally:
        agent.client.close()
