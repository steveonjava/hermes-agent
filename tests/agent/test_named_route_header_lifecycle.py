"""Selected named headers must survive real SDK credential rebuilds, offline."""
import json
import time
import pytest
from hermes_cli.config import atomic_config_replace
from tests.agent import test_retained_named_routes as route_fixtures

offline_context_metadata = route_fixtures.offline_context_metadata


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("status_code", [401, 429])
def test_named_fallback_rotation_preserves_selected_headers(tmp_path, monkeypatch, shared, status_code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    urls = {name: ("https://shared.example/v1" if shared else f"https://{name}.example/v1")
            for name in ("alpha", "beta")}
    entries = {name: {"api": urls[name], "api_key": f"configured-{name}",
                     "transport": "chat_completions", "extra_headers": {
                         "X-Route": name, "Authorization": f"Bearer synthetic-{name}-header-only"}}
               for name in ("alpha", "beta")}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": entries})
    pools = {name: route_fixtures._write_route_pool(name, urls[name]) for name in entries}
    agent = _real_agent(requested_provider="custom:alpha", base_url=urls["alpha"],
                        fallback_model={"provider": "custom:beta", "model": "fallback-model",
                                        "api_mode": "chat_completions"})
    agent._credential_pool = pools["alpha"]
    assert agent._swap_credential(pools["alpha"].select(model=agent.model)) is not False
    primary_ok = agent.client.default_headers.get("Authorization") == entries["alpha"]["extra_headers"]["Authorization"]
    assert primary_ok, "Control primary must start with its selected credential header"
    assert agent._try_activate_fallback()
    before_ok = agent.client.default_headers.get("Authorization") == entries["beta"]["extra_headers"]["Authorization"]
    assert before_ok, "Named fallback must start with its selected credential header"
    assert agent._recover_with_credential_pool(status_code=status_code, has_retried_429=True,
        error_context={"reset_at": time.time() + 3600})[0]
    assert agent.requested_provider == "custom:beta" and agent._credential_pool.provider == "beta"
    headers = agent.client.default_headers
    identity_ok = headers.get("X-Route") == "beta"
    credential_ok = headers.get("Authorization") == entries["beta"]["extra_headers"]["Authorization"]
    print(json.dumps({"shared_endpoint": shared, "status_code": status_code,
                      "selected_provider": "beta", "header_route_marker": headers.get("X-Route"),
                      "credential_header_is_selected": credential_ok}))
    isolated = identity_ok and credential_ok
    assert isolated, "SDK rebuild must retain selected named provider's credential header"
    assert agent._restore_primary_runtime()
    restored = agent.client.default_headers.get("X-Route") == "alpha"
    credential_restored = agent.client.default_headers.get("Authorization") == entries["alpha"]["extra_headers"]["Authorization"]
    assert restored and credential_restored, "Primary restoration must restore selected headers"
    assert agent.requested_provider == "custom:alpha"
    assert agent._credential_pool.provider == "alpha"
    agent.client.close()


def _real_agent(**kwargs):
    from unittest.mock import patch
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=128_000),
    ):
        return AIAgent(
            model="primary-model", api_key="synthetic-boot-key", provider="custom",
            quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs,
        )


@pytest.mark.parametrize("stage", ["constructor", "switch"])
def test_real_sdk_named_identity_at_public_entry(tmp_path, monkeypatch, stage):
    from openai import OpenAI

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    agent = _real_agent(base_url=endpoint, requested_provider="custom:beta" if stage == "constructor" else "custom:alpha")
    try:
        if stage == "switch":
            agent.switch_model("next-model", "custom:beta", api_key="synthetic-switch-key",
                               base_url=endpoint, api_mode="chat_completions")
        assert isinstance(agent.client, OpenAI)
        selected = agent.client.default_headers.get("X-Route") == "beta"
        assert selected, "Public lifecycle must select named headers rather than first URL match"
    finally:
        agent.client.close()


@pytest.mark.parametrize(("legacy", "identity"), [
    (False, "custom:beta"), (False, "beta"), (False, "Beta Account"),
    (False, "beta-account"), (False, "custom:beta-account"),
    (True, "Beta Account"), (True, "beta-account"), (True, "custom:beta-account"),
])
@pytest.mark.parametrize("empty", [False, True])
def test_named_alias_empty_and_explicit_override(tmp_path, monkeypatch, legacy, identity, empty):
    from openai import OpenAI

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    entries = {
        "alpha": {"name": "Alpha", "api": endpoint, "extra_headers": {"X-Route": "alpha", "X-Old-Only": "synthetic"}},
        "beta": {"name": "Beta Account", "api": endpoint,
                 "extra_headers": {} if empty else {"X-Route": "beta"}},
    }
    config = {"providers": entries}
    if legacy:
        config = {"custom_providers": [dict(entry, provider_key=key, base_url=entry["api"])
                                       for key, entry in entries.items()]}
    atomic_config_replace(tmp_path / "config.yaml", config)
    agent = _real_agent(base_url="https://override.example/v1", requested_provider=identity)
    try:
        assert isinstance(agent.client, OpenAI)
        selected = agent.client.default_headers.get("X-Route") == (None if empty else "beta")
        assert selected, "Named identity must survive explicit URL/key overrides"
        agent.switch_model("next-model", "custom", requested_provider="custom:alpha",
                           api_key="synthetic-alpha", base_url=endpoint, api_mode="chat_completions")
        agent.switch_model("last-model", "custom", requested_provider=identity,
                           api_key="synthetic-explicit", base_url=endpoint, api_mode="chat_completions")
        selected = agent.client.default_headers.get("X-Route") == (None if empty else "beta")
        clean = "X-Old-Only" not in agent.client.default_headers
        assert selected and clean, "Swaps must clear stale sibling headers, including for empty entries"
        assert agent.api_key == "synthetic-explicit"
    finally:
        agent.client.close()


@pytest.mark.parametrize("identity", ["", "custom", "custom:beta"])
def test_generic_url_fallback_native_attribution_and_precedence(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://openrouter.ai/api/v1"
    atomic_config_replace(tmp_path / "config.yaml", {
        "model": {"default_headers": {"X-User-Only": "synthetic-user", "X-Route": "user"}},
        "providers": {name: {"api": endpoint, "extra_headers": {"X-Route": name}}
                      for name in ("alpha", "beta")},
    })
    agent = _real_agent(base_url=endpoint, requested_provider=identity)
    try:
        expected = "beta" if identity == "custom:beta" else "alpha"
        def check():
            headers = agent.client.default_headers
            precedence = headers.get("X-Route") == expected and headers.get("X-User-Only") == "synthetic-user"
            attribution = bool(headers.get("HTTP-Referer")) and bool(headers.get("X-Title"))
            assert precedence and attribution, "Keep provider > user > native header precedence"
        check()
        agent.switch_model("next-model", "custom", requested_provider=identity or "custom",
                           api_key="synthetic-next", base_url=endpoint, api_mode="chat_completions")
        check()
        assert agent._replace_primary_openai_client(reason="interrupt_recovery")
        check()
    finally:
        agent.client.close()


def test_named_env_refresh_keeps_selected_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": endpoint, "key_env": "SYNTHETIC_ROUTE_KEY", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    values = {"SYNTHETIC_ROUTE_KEY": "synthetic-boot-key"}
    monkeypatch.setattr("agent.credential_pool.get_env_prefer_dotenv", lambda key: values.get(key, ""))
    agent = _real_agent(base_url=endpoint, requested_provider="custom:beta")
    try:
        assert agent._try_refresh_env_client_credentials() is False
        values["SYNTHETIC_ROUTE_KEY"] = "synthetic-edited-key"
        assert agent._try_refresh_env_client_credentials() is True
        selected = agent.client.default_headers.get("X-Route") == "beta"
        assert selected, "Environment credential refresh must retain named header identity"
        assert agent.client.api_key == values["SYNTHETIC_ROUTE_KEY"]
    finally:
        agent.client.close()


@pytest.mark.parametrize("identity", ["custom:beta", "beta", "custom:beta-account"])
def test_explicit_entry_list_uses_native_identity(identity):
    from hermes_cli.config_providers import get_custom_provider_extra_headers

    entries = [{"provider_key": name, "name": name + " Account", "base_url": "https://shared.example/v1",
                "extra_headers": {"X-Route": name}} for name in ("alpha", "beta")]
    headers = get_custom_provider_extra_headers("https://override.example/v1", entries, requested_provider=identity)
    selected = headers.get("X-Route") == "beta"
    assert selected, "Explicit entry lists must preserve stored key and display aliases"


@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse"])
def test_non_openai_header_exclusion(tmp_path, monkeypatch, wire):
    from types import SimpleNamespace
    from agent.client_lifecycle import ClientLifecycleMixin

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "beta": {"api": endpoint, "extra_headers": {"X-Route": "beta"}},
    }})
    agent = SimpleNamespace(provider="custom", requested_provider="custom:beta", api_mode=wire,
                            _client_kwargs={}, _apply_user_default_headers=lambda: None)
    ClientLifecycleMixin._apply_client_headers_for_base_url(agent, endpoint)
    assert "default_headers" not in agent._client_kwargs


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_beta_primary_restores_after_alpha_fallback(tmp_path, monkeypatch, shared, empty):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    urls = {name: ("https://shared.example/v1" if shared else f"https://{name}.example/v1")
            for name in ("alpha", "beta")}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": urls[name], "api_key": f"synthetic-{name}", "transport": "chat_completions",
               "extra_headers": {} if name == "beta" and empty else {"X-Route": name}}
        for name in urls
    }})
    pools = {name: route_fixtures._write_route_pool(name, urls[name]) for name in urls}
    agent = _real_agent(base_url=urls["beta"], requested_provider="custom:beta",
                        credential_pool=pools["beta"], fallback_model={
                            "provider": "custom:alpha", "model": "fallback-model"})
    expected = None if empty else "beta"
    try:
        selected = agent.client.default_headers.get("X-Route") == expected
        assert selected, "Beta primary constructor must not borrow alpha headers"
        assert agent._recover_with_credential_pool(status_code=429, has_retried_429=True,
            error_context={"reset_at": time.time() + 3600})[0]
        selected = agent.client.default_headers.get("X-Route") == expected
        assert selected, "Beta rotation must retain empty or populated named headers"
        assert agent._try_activate_fallback()
        selected = agent.client.default_headers.get("X-Route") == "alpha"
        assert selected, "Alpha fallback must have alpha headers"
        assert agent._restore_primary_runtime()
        selected = agent.client.default_headers.get("X-Route") == expected
        assert selected, "Beta primary restore must not retain alpha headers"
        assert agent.requested_provider == "custom:beta"
        assert agent._credential_pool.provider == "beta"
    finally:
        agent.client.close()
