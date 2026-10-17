"""Named fallback resolution uses real config/runtime resolution, without API requests."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.config import atomic_config_replace
from hermes_cli.runtime_provider import resolve_runtime_provider
from tests.agent.test_primary_runtime_restore import _make_agent


@pytest.fixture(autouse=True)
def offline_context_metadata(monkeypatch):
    import agent.agent_runtime_helpers as runtime_helpers
    from agent import route_binding
    from hermes_cli import config
    from hermes_cli import runtime_provider
    import run_agent

    root = Path(__file__).resolve().parents[2]
    for module in (runtime_helpers, route_binding, config, runtime_provider, run_agent):
        assert Path(module.__file__).resolve().is_relative_to(root)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 128_000)
    def reject_network(*args, **kwargs):
        raise AssertionError("Provider contract tests must not open network connections")
    monkeypatch.setattr("socket.socket.connect", reject_network)


@pytest.mark.parametrize("overrides", [{}, {
    "model": "gpt-5.6-explicit", "base_url": "https://override.example/v1",
    "api_key": "explicit-test-key", "api_mode": "chat_completions",
    "capabilities": {"vision": False},
}, {"transport": "chat_completions"}])
def test_named_fallback_inherits_resolved_runtime(tmp_path, monkeypatch, overrides):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"relay": {
        "api": "https://relay.example/v1", "default_model": "gpt-5.6",
        "api_key": "configured-test-key", "transport": "codex_responses",
        "capabilities": {"openai_native_compaction": True, "vision": True},
    }}})
    agent = _make_agent(model="primary-model")
    entry = {"provider": "custom:relay", **overrides}
    expected = resolve_runtime_provider(
        requested=entry["provider"], target_model=entry.get("model"),
        explicit_base_url=entry.get("base_url"), explicit_api_key=entry.get("api_key"),
    )
    if "api_key" in overrides:
        assert expected["api_key"] == overrides["api_key"]
        assert expected["base_url"] == overrides["base_url"]
    # The direct binder also serves per-turn routes with no explicit model.
    from agent.route_binding import bind_route_entry
    with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
        assert bind_route_entry(agent, entry, entry["provider"], entry.get("model", ""))
    assert agent.provider == expected["provider"]
    assert agent.requested_provider == expected["requested_provider"]
    assert agent.model == expected["model"]
    assert agent.base_url.rstrip("/") == expected["base_url"].rstrip("/")
    assert agent.api_key == expected["api_key"]
    assert agent.api_mode == overrides.get("api_mode", overrides.get("transport", expected["api_mode"]))
    expected_caps = dict(overrides.get("capabilities", expected["capabilities"]))
    if overrides.get("transport") == "chat_completions":
        expected_caps["openai_native_compaction"] = False
    assert agent.capabilities == expected_caps


@pytest.mark.parametrize("wire", ["chat_completions", "codex_responses"])
@pytest.mark.parametrize("override", [False, True])
def test_named_fallback_constructs_client_with_configured_headers(tmp_path, monkeypatch, wire, override):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"relay": {
        "api": "https://relay.example/v1", "api_key": "configured-test-key",
        "transport": "codex_responses" if wire == "chat_completions" else "chat_completions",
        "extra_headers": {"X-Relay-Secret": "synthetic-secret", "X-Route": "synthetic-route"},
    }}})
    agent = _make_agent(model="primary-model")
    entry = {"provider": "custom:relay", "model": "gpt-5.6", "api_mode": wire}
    if override:
        entry.update(api_key="explicit-test-key", base_url="https://override.example/v1")
    from agent.route_binding import bind_route_entry
    with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
        assert bind_route_entry(agent, entry, entry["provider"], entry["model"])
    from openai import OpenAI
    assert isinstance(agent.client, OpenAI)
    assert agent.client.default_headers.get("X-Relay-Secret") == "synthetic-secret"
    assert agent.client.default_headers.get("X-Route") == "synthetic-route"
    assert agent.client.api_key == ("explicit-test-key" if override else "configured-test-key")
    assert str(agent.client.base_url).rstrip("/") == (
        "https://override.example/v1" if override else "https://relay.example/v1")
    assert agent.api_mode == wire
    assert agent.provider == "custom"
    assert agent.requested_provider == "custom:relay"
    agent.client.close()


def test_named_fallback_activation_keeps_inherited_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"relay": {
        "api": "https://relay.example/v1", "api_key": "configured-test-key",
        "transport": "codex_responses", "capabilities": {"openai_native_compaction": True},
    }}})
    agent = _make_agent(model="primary-model", fallback_model={
        "provider": "custom:relay", "model": "gpt-5.6",
    })
    with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
        assert agent._try_activate_fallback()
    assert agent.provider == "custom"
    assert agent.requested_provider == "custom:relay"
    assert agent.capabilities == {"openai_native_compaction": True}
    assert agent.runtime_capabilities == {"native_compaction": True}


@pytest.mark.parametrize("legacy", [False, True])
def test_switch_snapshot_restore_uses_requested_pool_identity(tmp_path, monkeypatch, legacy):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "api_key": name + "-test-key"}
        for name in ("alpha", "beta")
    }})
    from agent.credential_pool import load_pool
    alpha_pool = load_pool("custom:alpha")
    beta_pool = load_pool("custom:beta")
    assert alpha_pool.select().runtime_api_key == "alpha-test-key"
    assert beta_pool.select().runtime_api_key == "beta-test-key"
    provider = "custom" if legacy else "custom:alpha"
    agent = _make_agent(model="gpt-5.1", provider=provider,
                        requested_provider="custom:alpha", base_url=endpoint)
    agent._credential_pool = alpha_pool
    switch_kwargs = {"requested_provider": "custom:beta"} if legacy else {}
    with patch("agent.process_bootstrap.OpenAI"):
        agent.switch_model(
            "gpt-5.6", "custom" if legacy else "custom:beta",
            api_key="beta-test-key", base_url=endpoint, api_mode="codex_responses",
            provider_capabilities={"openai_native_compaction": True}, **switch_kwargs,
        )
    assert agent.requested_provider == "custom:beta"
    assert agent._primary_runtime["requested_provider"] == "custom:beta"
    assert agent._credential_pool.provider in {"beta", "custom:beta"}
    assert agent._credential_pool.select().runtime_api_key == "beta-test-key"
    agent._credential_pool = alpha_pool
    agent.model, agent.provider, agent.requested_provider = "fallback-model", "custom", "custom:alpha"
    agent._fallback_activated = True
    with patch("agent.process_bootstrap.OpenAI"):
        assert agent._restore_primary_runtime()
    assert agent.provider == ("custom" if legacy else "custom:beta")
    assert agent.requested_provider == "custom:beta"
    assert agent._credential_pool.provider in {"beta", "custom:beta"}
    assert agent.api_key == "beta-test-key"
    assert agent._client_kwargs.get("default_headers") == agent._primary_runtime["client_kwargs"].get("default_headers")
    assert agent.capabilities == {"openai_native_compaction": True}


def _write_route_pool(provider, endpoint):
    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli.auth import write_credential_pool

    rows = [PooledCredential(
        provider=provider, id=f"{provider}-{i}", label=f"synthetic-{i}",
        source="manual", auth_type="api_key", priority=i,
        access_token=f"synthetic-{provider}-key-{i}", base_url=endpoint,
    ) for i in range(2)]
    write_credential_pool(provider, [row.to_dict() for row in rows])
    return load_pool(provider)


@pytest.mark.parametrize("initial", [None, "custom", "alpha", "beta"])
@pytest.mark.parametrize("overrides", [{}, {"api_key": "explicit-test-key"},
                                      {"base_url": "https://override.example/v1"}])
def test_named_fallback_scopes_real_pool_and_preserves_rotation(tmp_path, monkeypatch, initial, overrides):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "api_key": f"configured-{name}-key",
               "transport": "codex_responses", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    pools = {name: _write_route_pool(name, endpoint) for name in ("custom", "alpha", "beta")}
    from hermes_cli.auth import read_credential_pool
    from agent.credential_pool import load_pool
    from agent.route_binding import bind_route_entry
    agent = _make_agent(model="primary-model")
    agent._credential_pool = pools.get(initial)
    entry = {"provider": "custom:beta", "model": "gpt-5.6", "api_mode": "chat_completions", **overrides}
    untouched = {name: read_credential_pool(name) for name in ("custom", "alpha")}
    with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
        assert bind_route_entry(agent, entry, entry["provider"], entry["model"])
    assert agent.provider == "custom" and agent.requested_provider == "custom:beta"
    assert agent.api_mode == "chat_completions"
    assert agent.client.default_headers["X-Route"] == "beta"
    if overrides:
        assert agent._credential_pool is None
        assert agent._credential_pool_entry_id is None
        assert agent.api_key == overrides.get("api_key", "configured-beta-key")
        assert agent.base_url.rstrip("/") == overrides.get("base_url", endpoint)
        agent.client.close()
        return
    assert agent._credential_pool.provider == "beta"
    if initial == "beta":
        assert agent._credential_pool is pools["beta"]
    assert agent.api_key == "synthetic-beta-key-0"
    assert agent._credential_pool_entry_id == "beta-0"
    import time
    reset_at = time.time() + 3600
    recovered, _ = agent._recover_with_credential_pool(
        status_code=429, has_retried_429=True, error_context={"reset_at": reset_at})
    assert recovered
    assert agent.api_key == "synthetic-beta-key-1"
    assert agent._credential_pool_entry_id == "beta-1"
    assert agent.client.default_headers.get("X-Route") == "beta"
    pool = agent._credential_pool
    benched = next(e for e in pool.entries() if e.id == "beta-0")
    cooldown = benched.last_error_reset_at
    assert cooldown is not None
    with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
        assert bind_route_entry(agent, entry, entry["provider"], entry["model"])
    assert agent._credential_pool is pool
    assert agent.api_key == "synthetic-beta-key-1"
    assert agent._credential_pool_entry_id == "beta-1"
    assert agent.client.default_headers.get("X-Route") == "beta"
    assert next(e for e in pool.entries() if e.id == "beta-0").last_error_reset_at == cooldown
    assert load_pool("beta").select().id == "beta-1"
    for name, rows in untouched.items():
        assert read_credential_pool(name) == rows
    agent.client.close()


def test_same_provider_switch_keeps_pool_when_legacy_identity_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://relay.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"relay": {
        "api": endpoint, "api_key": "configured-test-key",
    }}})
    from agent.credential_pool import load_pool

    pool = load_pool("custom:relay")
    agent = _make_agent(model="gpt-5.1", provider="custom:relay", base_url=endpoint)
    agent.requested_provider = ""
    agent._credential_pool = pool
    with patch("agent.process_bootstrap.OpenAI"):
        agent.switch_model("gpt-5.6", "custom:relay", api_key="configured-test-key",
                           base_url=endpoint, api_mode="chat_completions")
    assert agent._credential_pool is pool
    assert agent.requested_provider == "custom:relay"
