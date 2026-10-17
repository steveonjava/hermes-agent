"""Actual ACP selector, constructor and persisted-route authority boundaries."""
import json
from types import SimpleNamespace

import pytest

from acp_adapter.session import SessionManager
from hermes_cli.runtime_binding import RuntimeBinding, runtime_binding_state


@pytest.fixture
def constructor_env(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *a: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **kw: None)
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: calls.append(kw) or SimpleNamespace(**kw))
    manager = SessionManager(db=None)
    monkeypatch.setattr(manager, "_get_db", lambda: None)
    return manager, calls


@pytest.mark.parametrize("policy", ["native", "route_override", "key_override"])
def test_actual_constructor_preserves_known_authority(monkeypatch, constructor_env, policy):
    manager, calls = constructor_env
    binding = RuntimeBinding("custom", "custom:beta", policy)
    seen = []
    def resolve(**kw):
        seen.append(kw)
        return {"provider": "custom", "requested_provider": "custom:beta", "runtime_binding": binding,
                "api_key": "synthetic-selected", "base_url": "https://selected.test", "credential_pool": None}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    manager._make_agent(session_id="s", cwd=".", model="m", requested_provider="custom:beta",
                        base_url="https://selected.test", api_key="synthetic-selected", runtime_binding=binding,
                        enabled_toolsets=["hermes-acp"])
    assert seen[0]["runtime_binding"] == binding
    assert seen[0]["explicit_api_key"] == "synthetic-selected"
    assert seen[0]["explicit_base_url"] == "https://selected.test"
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["requested_provider"] == "custom:beta"
    assert calls[0]["api_key"] == "synthetic-selected"


def test_actual_acp_switch_hands_result_to_real_constructor(monkeypatch, constructor_env):
    from acp_adapter.server import HermesACPAgent
    manager, calls = constructor_env
    binding = RuntimeBinding("custom", "custom:beta", "native")
    result = SimpleNamespace(success=True, target_provider="custom", new_model="m", api_key="synthetic-selected",
        base_url="https://selected.test", api_mode="chat_completions", runtime_binding=binding)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "custom", "requested_provider": kw.get("requested"), "runtime_binding": kw.get("runtime_binding"),
        "api_key": kw.get("explicit_api_key"), "base_url": kw.get("explicit_base_url")})
    monkeypatch.setattr(manager, "save_session", lambda sid: None)
    state = SimpleNamespace(session_id="s", cwd=".", model="old", agent=SimpleNamespace(
        provider="custom", base_url="https://old.test", api_key="synthetic-old", enabled_toolsets=["hermes-acp"]))
    HermesACPAgent(session_manager=manager)._switch_model(state, "m")
    assert calls[0]["runtime_binding"] == binding
    assert calls[0]["requested_provider"] == "custom:beta"
    assert calls[0]["api_key"] == "synthetic-selected"
    assert calls[0]["base_url"] == "https://selected.test"


@pytest.mark.parametrize("bad", [None, {}, {"version": 99}])
def test_actual_restore_rejects_present_invalid_without_rebuild(monkeypatch, bad):
    row = {"source": "acp", "model": "m", "model_config": json.dumps({"runtime_binding": bad})}
    db = SimpleNamespace(get_session=lambda sid: row, get_messages_as_conversation=lambda *a, **kw: [])
    manager = SessionManager(db=db)
    monkeypatch.setattr(manager, "_make_agent", lambda **kw: pytest.fail("invalid route must not rebuild"))
    with pytest.raises(ValueError, match="runtime binding"):
        manager.get_session("s")


def test_actual_restore_missing_pin_never_rebuilds(monkeypatch):
    binding = RuntimeBinding("custom", "custom:beta", "key_override")
    row = {"source": "acp", "model": "m", "model_config": json.dumps({
        "provider": "custom:beta", "runtime_binding": runtime_binding_state(binding)})}
    db = SimpleNamespace(get_session=lambda sid: row, get_messages_as_conversation=lambda *a, **kw: [])
    manager = SessionManager(db=db)
    monkeypatch.setattr(manager, "_make_agent", lambda **kw: pytest.fail("missing pin must not rebuild"))
    with pytest.raises(ValueError, match="[Rr]e-supply"):
        manager.get_session("s")


@pytest.mark.parametrize("policy", ["native", "route_override", None])
def test_actual_persist_restore_nonsecret_route(monkeypatch, tmp_path, policy):
    from hermes_state import SessionDB
    from acp_adapter.session import SessionState
    binding = RuntimeBinding("custom", "custom:beta", policy) if policy else None
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db)
    monkeypatch.setattr(manager, "_schedule_git_metadata", lambda *a: None)
    agent = SimpleNamespace(provider="custom", requested_provider="custom:beta", base_url="https://selected.test",
        api_mode="chat_completions", runtime_binding=binding, api_key="synthetic-private")
    state = SessionState(session_id="s", agent=agent, cwd=".", model="m", history=[{"role": "user", "content": "hi"}])
    manager._persist(state)
    row = db.get_session("s")
    meta = json.loads(row["model_config"])
    assert "synthetic-private" not in row["model_config"]
    if binding:
        assert meta["runtime_binding"] == runtime_binding_state(binding)
        assert meta["provider"] == "custom:beta"
    else:
        assert "runtime_binding" not in meta
    seen = []
    monkeypatch.setattr(manager, "_make_agent", lambda **kw: seen.append(kw) or agent)
    assert manager.get_session("s").agent is agent
    assert seen[0].get("runtime_binding") == binding
    db.close()


@pytest.mark.parametrize("policy", ["key_override", "route_override"])
def test_actual_constructor_never_default_falls_back(monkeypatch, constructor_env, policy):
    manager, calls = constructor_env
    binding = RuntimeBinding("custom", "custom:beta", policy)
    def resolve(**kw):
        raise ValueError("selected source unavailable")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    with pytest.raises(ValueError, match="selected source unavailable"):
        manager._make_agent(session_id="s", cwd=".", requested_provider="custom:beta",
                            runtime_binding=binding, api_key="synthetic-pin", enabled_toolsets=[])
    assert not calls


def test_acp_keep_endpoint_marks_named_override(monkeypatch, constructor_env):
    from acp_adapter.server import HermesACPAgent
    manager, calls = constructor_env
    binding = RuntimeBinding("custom", "custom:beta", "native")
    result = SimpleNamespace(success=True, target_provider="custom", new_model="m", api_key="synthetic-selected",
        base_url="https://selected.test", api_mode="chat_completions", runtime_binding=binding)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    def resolve(**kw):
        return {"provider": "custom", "requested_provider": kw["requested"], "runtime_binding": kw["runtime_binding"],
                "api_key": kw["explicit_api_key"], "base_url": kw["explicit_base_url"]}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr(manager, "save_session", lambda sid: None)
    state = SimpleNamespace(session_id="s", cwd=".", model="old", agent=SimpleNamespace(
        provider="custom", base_url="https://alternate.test", api_mode="chat_completions", api_key="synthetic-old",
        enabled_toolsets=[]))
    HermesACPAgent(session_manager=manager)._switch_model(state, "m", keep_endpoint=True)
    assert calls[0]["runtime_binding"].credential_policy == "route_override"
    assert calls[0]["runtime_binding"].requested_provider == "custom:beta"
    assert calls[0]["base_url"] == "https://alternate.test"


def test_actual_restore_selected_failure_cannot_resume_as_new(monkeypatch):
    import asyncio
    from acp_adapter.server import HermesACPAgent
    binding = RuntimeBinding("custom", "custom:beta", "route_override")
    row = {"source": "acp", "model": "m", "model_config": json.dumps({
        "provider": "custom:beta", "runtime_binding": runtime_binding_state(binding)})}
    db = SimpleNamespace(get_session=lambda sid: row, get_messages_as_conversation=lambda *a, **kw: [])
    manager = SessionManager(db=db)
    def fail(**kw):
        raise ValueError("selected route unavailable")
    monkeypatch.setattr(manager, "_make_agent", fail)
    monkeypatch.setattr(manager, "create_session", lambda **kw: pytest.fail("must not create default session"))
    with pytest.raises(ValueError, match="selected route unavailable"):
        asyncio.run(HermesACPAgent(session_manager=manager).resume_session(".", "s"))


def test_constructor_missing_pin_fails_before_resolver(monkeypatch, constructor_env):
    manager, calls = constructor_env
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: pytest.fail("missing pin resolves"))
    with pytest.raises(ValueError, match="[Rr]e-supply"):
        manager._make_agent(session_id="s", cwd=".", runtime_binding=RuntimeBinding("custom", "custom:beta", "key_override"))
    assert not calls
