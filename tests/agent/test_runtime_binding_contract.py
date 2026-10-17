"""Public runtime-binding lifecycle contract; scenario IDs follow approved matrix."""
import pytest

from hermes_cli.config import atomic_config_replace
from tests.agent import test_named_route_header_lifecycle as headers


offline_context_metadata = headers.offline_context_metadata


@pytest.mark.parametrize("destination", ["https://beta.example/v1", "https://unknown.example/v1"])
def test_I01_I02_implicit_destination_drops_named_authority(tmp_path, monkeypatch, destination):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": f"https://{name}.example/v1", "extra_headers": {"X-Route": name}}
        for name in ("alpha", "beta")
    }})
    agent = headers._real_agent(base_url="https://alpha.example/v1", requested_provider="custom:alpha")
    try:
        agent.switch_model("next", "custom", api_key="synthetic-next", base_url=destination,
                           api_mode="chat_completions")
        assert agent.requested_provider == "custom"
        assert agent.client.default_headers.get("X-Route") != "alpha"
    finally:
        agent.client.close()


@pytest.mark.parametrize("identity", ["custom:missing", "CUSTOM:missing", "  CUSTOM:missing  "])
def test_I07_normalized_unknown_namespace_no_header_borrow(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://shared.example/v1"
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": endpoint, "extra_headers": {"X-Route": "alpha"}}
    }})
    agent = headers._real_agent(base_url=endpoint, requested_provider=identity)
    try:
        assert "X-Route" not in agent.client.default_headers
    finally:
        agent.client.close()



@pytest.mark.parametrize("policy", ["native", "named_nonpool", "route_override", "key_override"])
def test_L01_L02_binding_constructor_switch_restore(tmp_path, monkeypatch, policy):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "extra_headers": {"X-Route": "alpha"}}
    }})
    binding = RuntimeBinding("custom", "custom:alpha", policy)
    agent = headers._real_agent(base_url="https://alpha.example/v1", requested_provider="custom:alpha",
                                runtime_binding=binding)
    try:
        assert agent.runtime_binding is binding
        assert agent._primary_runtime["runtime_binding"] is binding
        agent.switch_model("next", "custom", requested_provider="custom:alpha", runtime_binding=binding,
                           api_key="synthetic-next", base_url="https://alpha.example/v1", api_mode="chat_completions")
        assert agent._primary_runtime["runtime_binding"] is binding
        assert agent._try_refresh_env_client_credentials() is False
        if policy != "native":
            assert agent._credential_pool is None
        agent._fallback_activated = True
        assert agent._restore_primary_runtime()
        assert agent.runtime_binding is binding
    finally:
        agent.client.close()


@pytest.mark.parametrize("policy", ["key_override", "route_override", "named_nonpool"])
def test_P03_P04_detached_recovery_and_refresh(tmp_path, monkeypatch, policy):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SYNTHETIC_ROUTE_KEY", "synthetic-unrelated")
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "key_env": "SYNTHETIC_ROUTE_KEY"}
    }})
    agent = headers._real_agent(base_url="https://alpha.example/v1", requested_provider="custom:alpha",
                                runtime_binding=RuntimeBinding("custom", "custom:alpha", policy))
    try:
        assert agent._recover_with_credential_pool(status_code=401, has_retried_429=False)[0] is False
        if policy == "key_override":
            assert agent._try_refresh_env_client_credentials() is False
            assert agent.api_key == "synthetic-boot-key"
    finally:
        agent.client.close()


def test_L08_contradiction_rejected_before_mutation(tmp_path, monkeypatch):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = headers._real_agent(base_url="https://alpha.example/v1")
    before = (agent.model, agent.client, agent._primary_runtime)
    try:
        with pytest.raises(ValueError, match="binding"):
            agent.switch_model("next", "custom", requested_provider="custom:beta",
                               runtime_binding=RuntimeBinding("openrouter", "openrouter"),
                               api_key="synthetic", base_url="https://beta.example/v1")
        assert (agent.model, agent.client, agent._primary_runtime) == before
    finally:
        agent.client.close()


@pytest.mark.parametrize(("hint", "policy"), [({}, "native"),
    ({"base_url": "https://override.example/v1"}, "route_override"),
    ({"api_key": "synthetic-pin"}, "key_override")])
def test_P08_fallback_binding_and_restore(tmp_path, monkeypatch, hint, policy):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        name: {"api": f"https://{name}.example/v1", "api_key": "synthetic-configured"}
        for name in ("alpha", "beta")
    }})
    binding = RuntimeBinding("custom", "custom:alpha", "key_override")
    agent = headers._real_agent(base_url="https://alpha.example/v1", requested_provider="custom:alpha",
        runtime_binding=binding, fallback_model={"provider": "custom:beta", "model": "fallback", **hint})
    try:
        for _ in range(2):
            assert agent._try_activate_fallback()
            assert agent.runtime_binding == RuntimeBinding("custom", "custom:beta", policy)
            assert agent._restore_primary_runtime()
            assert agent.runtime_binding is binding
            assert agent._credential_pool is None
    finally:
        agent.client.close()


@pytest.mark.parametrize("identity", ["custom:missing", " CUSTOM:missing "])
def test_I07_unknown_namespace_never_loads_url_pool(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://shared.example/v1", "api_key": "synthetic-configured"}
    }})
    agent = headers._real_agent(base_url="https://shared.example/v1")
    called = []
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda *a, **k: called.append(True))
    try:
        agent.switch_model("next", "custom", requested_provider=identity, api_key="synthetic",
                           base_url="https://shared.example/v1", api_mode="chat_completions")
        assert not called
    finally:
        agent.client.close()


def test_P04_route_override_refresh_keeps_destination(tmp_path, monkeypatch):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "key_env": "SYNTHETIC_ROUTE_KEY"}
    }})
    values = {"SYNTHETIC_ROUTE_KEY": "synthetic-boot-key"}
    monkeypatch.setattr("agent.credential_pool.get_env_prefer_dotenv", lambda key: values.get(key, ""))
    agent = headers._real_agent(base_url="https://override.example/v1", requested_provider="custom:alpha",
                                runtime_binding=RuntimeBinding("custom", "custom:alpha", "route_override"))
    try:
        assert agent._try_refresh_env_client_credentials() is False
        values["SYNTHETIC_ROUTE_KEY"] = "synthetic-updated"
        assert agent._try_refresh_env_client_credentials()
        assert agent.base_url == "https://override.example/v1"
        assert agent.api_key == "synthetic-updated"
        assert agent._credential_pool is None
    finally:
        agent.client.close()


@pytest.mark.parametrize("value", [{"version": 2}, {"version": True}, "synthetic-secret-invalid"])
def test_L08_invalid_binding_no_public_mutation(tmp_path, monkeypatch, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = headers._real_agent(base_url="https://alpha.example/v1")
    before = (agent.client, agent.model, agent._primary_runtime)
    try:
        with pytest.raises(ValueError, match="Invalid runtime binding metadata") as caught:
            agent.switch_model("next", "custom", runtime_binding=value)
        assert "synthetic-secret-invalid" not in str(caught.value)
        assert (agent.client, agent.model, agent._primary_runtime) == before
    finally:
        agent.client.close()


def test_L04_nous_wire_forwards_binding(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.runtime_binding import RuntimeBinding
    from agent.nous_wire import apply_pending_wire_switch

    binding = RuntimeBinding("nous", "nous")
    agent = SimpleNamespace(_nous_wire_pending=("chat_completions", "test"), model="m", api_key="synthetic",
                            base_url="https://inference-api.nousresearch.com/v1", api_mode="codex_responses",
                            runtime_binding=binding, requested_provider="nous")
    seen = {}
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *a, **kw: seen.update(kw))
    assert apply_pending_wire_switch(agent)
    assert seen["runtime_binding"] is binding
    assert seen["requested_provider"] == "nous"


@pytest.mark.parametrize("failure", ["client", "compressor"])
def test_L07_real_public_switch_rollback(tmp_path, monkeypatch, failure):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    binding = RuntimeBinding("custom", "custom", "key_override")
    agent = headers._real_agent(base_url="https://alpha.example/v1", runtime_binding=binding)
    before = (agent.runtime_binding, agent.client, agent._primary_runtime, agent._credential_pool)
    def fail(*args, **kwargs):
        raise ValueError("synthetic build failure")
    try:
        if failure == "client":
            monkeypatch.setattr(agent, "_create_openai_client", fail)
        else:
            monkeypatch.setattr(agent.context_compressor, "update_model", fail)
        with pytest.raises(ValueError):
            agent.switch_model("next", "custom", api_key="synthetic-next", base_url="https://beta.example/v1",
                               runtime_binding=RuntimeBinding("custom", "custom", "route_override"))
        assert (agent.runtime_binding, agent.client, agent._primary_runtime, agent._credential_pool) == before
    finally:
        agent.client.close()


def test_L09_standing_voice_metadata():
    from types import SimpleNamespace
    from hermes_cli.runtime_binding import RuntimeBinding
    from agent.voice_turn_route import _SessionRuntimeView

    binding = RuntimeBinding("custom", "custom:alpha", "key_override")
    agent = SimpleNamespace(requested_provider="custom:beta", runtime_binding=None)
    view = _SessionRuntimeView(agent, {"snapshot": {
        "requested_provider": "custom:alpha", "runtime_binding": binding}})
    assert view.requested_provider == "custom:alpha"
    assert view.runtime_binding is binding


def test_I07_header_helper_normalizes_namespace():
    from hermes_cli.config_providers import get_custom_provider_extra_headers

    entries = [{"provider_key": "alpha", "base_url": "https://shared.example/v1",
                "extra_headers": {"X-Route": "alpha"}}]
    assert get_custom_provider_extra_headers("https://shared.example/v1", entries,
                                             requested_provider="  CUSTOM:missing  ") == {}


def test_L08_fallback_pinned_missing_key_cannot_borrow(tmp_path, monkeypatch):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "beta": {"api": "https://beta.example/v1", "api_key": "synthetic-unrelated"}}})
    agent = headers._real_agent(base_url="https://primary.example/v1", requested_provider="custom",
        fallback_model={"provider": "custom:beta", "model": "fallback", "runtime_binding":
                        RuntimeBinding("custom", "custom:beta", "key_override")})
    try:
        before = agent.client
        assert agent._try_activate_fallback() is False
        assert agent.client is before
        assert agent.requested_provider == "custom"
    finally:
        agent.client.close()


@pytest.mark.parametrize(("destination", "expected"), [
    ("https://ALPHA.example:443/v1/", "custom:alpha"),
    ("https://alpha.example/V1", "custom"),
    ("https://alpha.example/v1?account=beta", "custom"),
])
def test_I12_native_url_equivalence_preserves_only_same_route(tmp_path, monkeypatch, destination, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "extra_headers": {"X-Route": "alpha"}}}})
    agent = headers._real_agent(base_url="https://alpha.example/v1", requested_provider="custom:alpha")
    try:
        agent.switch_model("next", "custom", api_key="synthetic-next", base_url=destination,
                           api_mode="chat_completions")
        assert agent.requested_provider == expected
        assert (agent.client.default_headers.get("X-Route") == "alpha") == (expected == "custom:alpha")
    finally:
        agent.client.close()


def test_L08_builtin_fallback_pin_validated_before_client(tmp_path, monkeypatch):
    from hermes_cli.runtime_binding import RuntimeBinding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    agent = headers._real_agent(base_url="https://primary.example/v1", requested_provider="custom",
        fallback_model={"provider": "openrouter", "model": "fallback", "runtime_binding":
                        RuntimeBinding("openrouter", "openrouter", "key_override")})
    consulted = []
    def forbidden(*args, **kwargs):
        consulted.append(True)
        raise AssertionError("Missing pinned key reached credential/client resolver")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", forbidden)
    try:
        assert agent._try_activate_fallback() is False
        assert not consulted
    finally:
        agent.client.close()
