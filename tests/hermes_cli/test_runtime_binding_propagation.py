"""Native producer authority controls, using isolated configured routes."""
import pytest

from hermes_cli.config import atomic_config_replace
from hermes_cli.runtime_binding import RuntimeBinding
from hermes_cli.runtime_provider import resolve_runtime_provider


@pytest.mark.parametrize(("key", "url", "policy"), [
    (None, None, "native"), ("synthetic-pin", None, "key_override"),
    (None, "https://alternate.example/v1", "route_override"),
])
def test_R02_I03_I04_intake_policy(tmp_path, monkeypatch, key, url, policy):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}
    }})
    runtime = resolve_runtime_provider(requested="custom:alpha", explicit_api_key=key, explicit_base_url=url)
    assert runtime["runtime_binding"] == RuntimeBinding("custom", "custom:alpha", policy)


def test_R02_forwarded_key_keeps_native_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}
    }})
    binding = RuntimeBinding("custom", "custom:alpha")
    runtime = resolve_runtime_provider(requested="Alpha", explicit_api_key="synthetic-resolved",
                                       runtime_binding=binding)
    assert runtime["runtime_binding"] is binding


def test_R02_selector_carries_native_binding(tmp_path, monkeypatch):
    from hermes_cli.model_switch import switch_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}
    }})
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **kw: {"accepted": True})
    result = switch_model("next", "custom", "old", "https://old.example/v1", explicit_provider="alpha", user_providers={"alpha": {
        "api": "https://alpha.example/v1", "api_key": "synthetic-configured"}})
    assert result.success
    assert result.runtime_binding == RuntimeBinding("custom", "custom:alpha")


@pytest.mark.parametrize("policy", ["route_override", "named_nonpool"])
def test_P04_policy_blocks_pool_before_resolution(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}
    }})
    def forbidden(*args, **kwargs):
        raise AssertionError("Detached policy consulted pool")
    monkeypatch.setattr("hermes_cli.runtime_provider._try_resolve_from_custom_pool", forbidden)
    runtime = resolve_runtime_provider(requested="custom:alpha", runtime_binding=RuntimeBinding(
        "custom", "custom:alpha", policy))
    assert runtime["runtime_binding"].credential_policy == policy


def test_I04_callable_pin_keeps_original_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}
    }})
    source = lambda: "synthetic-minted"
    runtime = resolve_runtime_provider(requested="custom:alpha", explicit_api_key=source)
    assert runtime["api_key"] is source
    assert runtime["runtime_binding"].credential_policy == "key_override"


def test_L08_pinned_missing_key_cannot_borrow(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}
    }})
    with pytest.raises(ValueError, match="re-supply"):
        resolve_runtime_provider(requested="custom:alpha", runtime_binding=RuntimeBinding(
            "custom", "custom:alpha", "key_override"))


@pytest.mark.parametrize("intent", ["key", "callable", "detached"])
def test_R02_direct_alias_intent_precedes_pool(tmp_path, monkeypatch, intent):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    def forbidden(*args, **kwargs):
        raise AssertionError("Alias consulted pool before honoring input policy")
    monkeypatch.setattr("hermes_cli.runtime_provider._try_resolve_from_custom_pool", forbidden)
    key = (lambda: "synthetic-token") if intent == "callable" else "synthetic-pin"
    binding = RuntimeBinding("custom", "custom", "named_nonpool") if intent == "detached" else None
    runtime = resolve_runtime_provider(requested="custom", explicit_api_key=key,
                                       explicit_base_url="https://alias.example/v1", runtime_binding=binding)
    assert runtime["api_key"] is key
    assert runtime["runtime_binding"].credential_policy == ("named_nonpool" if binding else "key_override")


def test_R02_alias_key_result_preserves_intent(tmp_path, monkeypatch):
    from hermes_cli.model_switch import switch_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"model_aliases": {"pinned": {
        "model": "synthetic-model", "provider": "custom", "base_url": "https://alias.example/v1",
        "api_key": "synthetic-alias-pin"}}})
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **kw: {"accepted": True})
    result = switch_model("pinned", "custom", "old", "https://old.example/v1", "synthetic-old")
    assert result.success
    assert result.api_key == "synthetic-alias-pin"
    assert result.runtime_binding == RuntimeBinding("custom", "custom", "key_override")


def test_R02_current_pinned_selector_carries_binding(tmp_path, monkeypatch):
    from hermes_cli.model_switch import switch_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {
        "alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-configured"}}})
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **kw: {"accepted": True})
    binding = RuntimeBinding("custom", "custom:alpha", "key_override")
    result = switch_model("next", "custom:alpha", "old", "https://alpha.example/v1", "synthetic-pin",
                          runtime_binding=binding)
    assert result.success
    assert result.api_key == "synthetic-pin"
    assert result.runtime_binding is binding


@pytest.mark.parametrize("kind", ["current", "configured", "alias-native"])
def test_R02_retained_producer_branches_always_carry_metadata(tmp_path, monkeypatch, kind):
    from hermes_cli.model_switch import switch_model

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {}
    if kind == "configured":
        config = {"providers": {"alpha": {"api": "https://alpha.example/v1", "api_key": "synthetic-config",
                                           "models": ["next"]}}}
    if kind == "alias-native":
        config = {"model_aliases": {"next": {"model": "synthetic-model", "provider": "custom",
                                             "base_url": "https://current.example/v1"}}}
    atomic_config_replace(tmp_path / "config.yaml", config)
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **kw: {"accepted": True})
    result = switch_model("next", "custom", "old", "https://current.example/v1", "synthetic-current",
                          user_providers=config.get("providers"))
    assert result.success
    assert result.runtime_binding == RuntimeBinding("custom", "custom:alpha" if kind == "configured" else "custom")


def test_P05_direct_alias_pool_source_is_native(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    pool = object()
    monkeypatch.setattr("hermes_cli.runtime_provider._try_resolve_from_custom_pool", lambda *a, **kw: {
        "provider": "custom", "api_key": "synthetic-pool", "base_url": "https://alias.example/v1",
        "api_mode": "chat_completions", "credential_pool": pool})
    runtime = resolve_runtime_provider(requested="custom", explicit_base_url="https://alias.example/v1")
    assert runtime["source"] == "direct-alias"
    assert runtime["credential_pool"] is pool
    assert runtime["runtime_binding"] == RuntimeBinding("custom", "custom")


@pytest.mark.parametrize("requested", ["auto", "openai"])
def test_R02_resolved_identity_is_consumable(tmp_path, monkeypatch, requested):
    from hermes_cli.runtime_provider_custom import _validate_runtime_binding

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-router")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai")
    atomic_config_replace(tmp_path / "config.yaml", {})
    runtime = resolve_runtime_provider(requested=requested)
    binding = runtime["runtime_binding"]
    _validate_runtime_binding(binding, runtime["provider"], runtime.get("requested_provider") or binding.requested_provider)


@pytest.mark.parametrize("identity", ["custom:missing", "  CUSTOM:missing  ", "unregistered-plugin"])
def test_I07_I10_resolver_identity_compatibility(tmp_path, monkeypatch, identity):
    from hermes_cli.auth import AuthError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    # Native resolver rejects unknown registrations; the public switch's direct route is separate.
    with pytest.raises(AuthError, match="Unknown provider"):
        resolve_runtime_provider(requested=identity, explicit_api_key="synthetic-pin",
                                 explicit_base_url="https://direct.example/v1")
