"""Explicit fallback overrides stay isolated while primary pool recovery survives."""
import json
import time
from unittest.mock import patch
import pytest
from hermes_cli.config import atomic_config_replace
from tests.agent.test_primary_runtime_restore import _make_agent
from tests.agent import test_retained_named_routes as route_fixtures


_write_route_pool = route_fixtures._write_route_pool
offline_context_metadata = route_fixtures.offline_context_metadata


@pytest.mark.parametrize("status_code", [401, 429])
@pytest.mark.parametrize("shared_endpoint", [False, True])
@pytest.mark.parametrize("override", [None, "api_key", "base_url", "both"])
def test_explicit_named_fallback_restores_primary_pool_and_rotation(tmp_path, monkeypatch, override, shared_endpoint, status_code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoints = {name: f"https://{name}.example/v1" for name in ("alpha", "beta")}
    if shared_endpoint:
        endpoints["beta"] = endpoints["alpha"]
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "api_key": f"configured-{name}-key",
               "transport": "chat_completions"}
        for name, endpoint in endpoints.items()
    }})
    pools = {name: _write_route_pool(name, endpoint) for name, endpoint in endpoints.items()}
    _write_route_pool("custom", endpoints["alpha"])
    from hermes_cli.auth import read_credential_pool
    untouched = {name: read_credential_pool(name) for name in ("custom", "beta")}
    fallback = {"provider": "custom:beta", "model": "fallback-model"}
    if override in ("api_key", "both"):
        fallback["api_key"] = "synthetic-explicit-key"
    if override in ("base_url", "both"):
        fallback["base_url"] = "https://override.example/v1"
    agent = _make_agent(model="primary-model", provider="custom",
                        requested_provider="custom:alpha", base_url=endpoints["alpha"],
                        fallback_model=fallback)
    agent._credential_pool = pools["alpha"]
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._swap_credential(pools["alpha"].select(model=agent.model)) is not False
        assert agent._credential_pool_entry_id == "alpha-0"
        assert agent._try_activate_fallback()
        assert agent.requested_provider == "custom:beta"
        if override:
            assert agent._credential_pool is None
            assert agent._credential_pool_entry_id is None
            fallback_key, fallback_url = agent.api_key, agent.base_url
            assert fallback_key == fallback.get("api_key", "configured-beta-key")
            assert fallback_url.rstrip("/") == fallback.get("base_url", endpoints["beta"])
            assert not agent._recover_with_credential_pool(
                status_code=status_code, has_retried_429=True)[0]
            assert (agent.api_key, agent.base_url) == (fallback_key, fallback_url)
        else:
            assert agent._credential_pool.provider == "beta"
        assert agent._restore_primary_runtime()
        restored_pool = getattr(agent._credential_pool, "provider", None)
        restored_entry = agent._credential_pool_entry_id
        rotated, _ = agent._recover_with_credential_pool(
            status_code=status_code, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600})
    assert {name: read_credential_pool(name) for name in untouched} == untouched
    failed = next(e for e in agent._credential_pool.entries() if e.id == "alpha-0")
    assert failed.last_error_code == status_code
    assert failed.last_error_reset_at is not None
    assert agent.base_url.rstrip("/") == endpoints["alpha"]
    assert agent.requested_provider == "custom:alpha"
    assert agent.api_key == "synthetic-alpha-key-1"
    result = {"override": override, "restored_pool": restored_pool,
              "restored_entry": restored_entry, "rotation_succeeded": rotated,
              "entry_after_rotation": agent._credential_pool_entry_id}
    print(json.dumps(result))
    assert result == {"override": override, "restored_pool": "alpha",
                      "restored_entry": "alpha-0", "rotation_succeeded": True,
                      "entry_after_rotation": "alpha-1"}


@pytest.mark.parametrize("override", [None, "api_key"])
@pytest.mark.parametrize("primary_alias", ["custom:alpha", "alpha", "Alpha Display", "custom:alpha-display"])
def test_selected_named_nonpool_primary_is_not_replaced(tmp_path, monkeypatch, override, primary_alias):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoints = {name: f"https://{name}.example/v1" for name in ("alpha", "beta")}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"name": "Alpha Display" if name == "alpha" else name,
               "api": url, "api_key": f"configured-{name}-key"}
        for name, url in endpoints.items()
    }})
    for name, url in endpoints.items():
        _write_route_pool(name, url)
    fallback = {"provider": "custom:beta", "model": "fallback-model"}
    if override:
        fallback[override] = "synthetic-explicit-fallback"
    agent = _make_agent(model="primary-model", requested_provider=primary_alias,
                        base_url=endpoints["alpha"], fallback_model=fallback)
    assert agent._credential_pool is None
    original_key = agent.api_key
    original_identity = agent.requested_provider
    from hermes_cli.runtime_provider import _get_named_custom_provider
    assert _get_named_custom_provider(primary_alias, resolve_credentials=False) is not None
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._try_activate_fallback()
        with patch("agent.credential_pool.load_pool", side_effect=AssertionError("selected nonpool runtime must not load a pool")) as load:
            assert agent._restore_primary_runtime()
        load.assert_not_called()
    assert agent.requested_provider == original_identity
    assert agent.base_url.rstrip("/") == endpoints["alpha"]
    assert agent.api_key == original_key
    assert agent._credential_pool is None
    assert agent._credential_pool_entry_id is None


@pytest.mark.parametrize("blocked_by", ["reset", "cooldown"])
def test_detached_fallback_waits_for_primary_recovery(tmp_path, monkeypatch, blocked_by):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoints = {name: f"https://{name}.example/v1" for name in ("alpha", "beta")}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": url, "api_key": f"configured-{name}-key"}
        for name, url in endpoints.items()
    }})
    pool = _write_route_pool("alpha", endpoints["alpha"])
    agent = _make_agent(model="primary-model", requested_provider="custom:alpha",
                        base_url=endpoints["alpha"], fallback_model={
                            "provider": "custom:beta", "model": "fallback-model",
                            "api_key": "synthetic-explicit-key"})
    agent._credential_pool = pool
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._swap_credential(pool.select(model=agent.model)) is not False
        if blocked_by == "reset":
            reset_at = time.time() + 3600
            # Exhaust all real primary entries, including the config-seeded credential.
            while pool.has_available(model=agent.model):
                row = pool.select(model=agent.model)
                pool.mark_exhausted_and_rotate(status_code=429, credential_id=row.id,
                                               error_context={"reset_at": reset_at})
        assert agent._try_activate_fallback()
        if blocked_by == "cooldown":
            agent._rate_limited_until = time.monotonic() + 3600
        assert not agent._restore_primary_runtime()
        assert agent._fallback_activated
        assert agent.requested_provider == "custom:beta"
        assert agent.api_key == "synthetic-explicit-key"
        assert agent._credential_pool is None
        assert agent._credential_pool_entry_id is None
        if blocked_by == "reset":
            rows = read_credential_pool("alpha")
            for row in rows:
                row.update(last_status="ok", last_status_at=None,
                           last_error_code=None, last_error_reset_at=None)
            write_credential_pool("alpha", rows, status_cleared_ids=[row["id"] for row in rows])
            assert load_pool("alpha").has_available(model="primary-model")
        else:
            agent._rate_limited_until = 0
        assert agent._restore_primary_runtime()
        assert agent._credential_pool.provider == "alpha"
        assert agent._credential_pool_entry_id == "alpha-0"
        assert agent.api_key == "synthetic-alpha-key-0"
        assert not agent._fallback_activated
        assert agent._fallback_index == 0
        # A later cycle must capture the primary again, not the detached fallback.
        assert agent._try_activate_fallback()
        assert agent._restore_primary_runtime()
        assert agent._credential_pool_entry_id == "alpha-0"


@pytest.mark.parametrize("provider", ["openai-api", "custom"])
def test_ordinary_nonpool_primary_none_to_none_keeps_native_lazy_behavior(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "beta": {"api": "https://beta.example/v1", "api_key": "synthetic-beta"}}})
    agent = _make_agent(model="primary-model", provider=provider,
                        requested_provider=provider, base_url="https://primary.example/v1",
                        fallback_model={"provider": "custom:beta", "model": "fallback-model",
                                        "api_key": "synthetic-explicit-fallback"})
    original_key = agent.api_key
    assert agent._credential_pool is None
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._try_activate_fallback()
        assert agent._credential_pool is None
        with patch("agent.credential_pool.load_pool", return_value=None) as load:
            assert agent._restore_primary_runtime()
        # Native None->None restoration does not discover a previously unattached pool.
        load.assert_not_called()
    assert agent.api_key == original_key
    assert agent._credential_pool is None
    assert agent._credential_pool_entry_id is None
