"""Offline contracts for transport-scoped trust, not a generic capability allowlist."""
from copy import deepcopy
from pathlib import Path

import pytest

from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_cli.config import atomic_config_replace

FLAGS = ("openai_native_compaction", "anthropic_tool_cache_marker", "anthropic_signed_thinking")
SECRET = "synthetic-credential-must-not-render"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from hermes_cli import config_providers, runtime_provider_custom
    from agent import route_binding
    root = Path(__file__).resolve().parents[2]
    for module in (config_providers, runtime_provider_custom, route_binding):
        assert Path(module.__file__).resolve().is_relative_to(root)
    def reject_network(*a, **k):
        raise AssertionError("No network in capability contracts")
    monkeypatch.setattr("socket.socket.connect", reject_network)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 128_000)


def save(tmp_path, monkeypatch, caps, wire="codex_responses", model=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = {"api": "https://relay.example/v1", "api_key": SECRET,
             "transport": wire, "default_model": "model", "capabilities": caps}
    if model is not None:
        entry["models"] = {"model": model}
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"relay": entry}})
    return entry


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [SECRET, 1, 0, None, [], {"secret": SECRET}])
def test_runtime_rejects_malformed_trust_value(tmp_path, monkeypatch, caplog, flag, value):
    save(tmp_path, monkeypatch, {flag: value})
    with pytest.raises(ValueError) as exc:
        resolve_runtime_provider(requested="custom:relay")
    assert flag in str(exc.value)
    assert SECRET not in str(exc.value) + caplog.text


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("wire", ["chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"])
def test_enabled_trust_requires_its_wire(tmp_path, monkeypatch, flag, wire):
    save(tmp_path, monkeypatch, {flag: True}, wire)
    allowed = "codex_responses" if flag == FLAGS[0] else "anthropic_messages"
    if wire == allowed:
        assert resolve_runtime_provider(requested="custom:relay")["capabilities"][flag] is True
    else:
        with pytest.raises(ValueError, match=flag):
            resolve_runtime_provider(requested="custom:relay")


@pytest.mark.parametrize("wire", ["openai-chat", "responses", "messages", "bedrock"])
def test_disabled_trust_and_arbitrary_generic_flags_survive(tmp_path, monkeypatch, wire):
    caps = {flag: False for flag in FLAGS}
    caps.update({"future.unregistered-capability": True, "vision": False})
    save(tmp_path, monkeypatch, caps, wire)
    assert resolve_runtime_provider(requested="custom:relay")["capabilities"] == caps


@pytest.mark.parametrize("flag,wire,url", [
    (FLAGS[0], "openai-responses", "https://relay.example/v1"),
    (FLAGS[1], "messages", "https://relay.example/v1"),
    (FLAGS[2], "anthropic-messages", "https://relay.example/v1"),
    (FLAGS[0], "", "https://api.openai.com/v1"),
    (FLAGS[1], "", "https://relay.example/anthropic"),
])
def test_canonical_and_inferred_runtime_wire(tmp_path, monkeypatch, flag, wire, url):
    entry = save(tmp_path, monkeypatch, {flag: True}, wire)
    entry["api"] = url
    atomic_config_replace(tmp_path / "config.yaml", {"providers": {"relay": entry}})
    assert resolve_runtime_provider(requested="custom:relay")["capabilities"][flag] is True


@pytest.mark.parametrize("value", [SECRET, True])
@pytest.mark.parametrize("path", ["modern", "legacy", "conversion", "migration"])
@pytest.mark.parametrize("flat", [False, True])
def test_config_paths_reject_before_mutation(tmp_path, monkeypatch, value, path, flat):
    from hermes_cli.config_providers import get_compatible_custom_providers, _custom_provider_entry_to_provider_config
    from hermes_cli.config_migrations import _migrate_to_12
    flag = FLAGS[1]
    entry = save(tmp_path, monkeypatch, {} if flat else {flag: value}, "responses",
                 {flag: value, "context_length": 12345, "description": "metadata"} if flat else None)
    legacy = {**entry, "name": "relay", "base_url": entry["api"], "api_mode": "responses"}
    config = {"providers": {"relay": entry}} if path == "modern" else {"_config_version": 11, "custom_providers": [legacy]}
    original = deepcopy(config)
    atomic_config_replace(tmp_path / "config.yaml", config)
    before = (tmp_path / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match=flag) as exc:
        if path == "conversion":
            _custom_provider_entry_to_provider_config(legacy)
        elif path == "migration":
            _migrate_to_12({"config_added": [], "warnings": []}, quiet=True)
        else:
            get_compatible_custom_providers(config)
    assert config == original
    assert (tmp_path / "config.yaml").read_bytes() == before
    assert SECRET not in str(exc.value)


@pytest.mark.parametrize("value", [SECRET, True])
def test_runtime_flat_model_trust_cannot_bypass_validation(tmp_path, monkeypatch, value):
    save(tmp_path, monkeypatch, {"future_flag": True}, "responses", {FLAGS[1]: value})
    with pytest.raises(ValueError, match=FLAGS[1]):
        resolve_runtime_provider(requested="custom:relay", target_model="model")


def test_runtime_flat_model_generic_metadata_is_preserved(tmp_path, monkeypatch):
    entry = save(tmp_path, monkeypatch, {"future_flag": True}, "messages",
                 {FLAGS[1]: True, "future_model_flag": False, "context_length": 12345, "description": "metadata"})
    before = deepcopy(entry)
    runtime = resolve_runtime_provider(requested="custom:relay", target_model="model")
    assert runtime["capabilities"] == {"future_flag": True, FLAGS[1]: True, "future_model_flag": False}
    assert entry == before


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("value", [True, SECRET])
def test_fallback_trust_policy_before_agent_mutation(tmp_path, monkeypatch, explicit, value):
    from agent.route_binding import bind_route_entry
    from tests.agent.test_primary_runtime_restore import _make_agent
    from unittest.mock import patch
    save(tmp_path, monkeypatch, {FLAGS[0]: True, "future_flag": True}, "responses")
    agent = _make_agent(model="primary-model")
    entry = {"transport": "openai-chat"}
    if explicit:
        entry["capabilities"] = {FLAGS[0]: value}
    before = dict(agent.__dict__)
    saved = (tmp_path / "config.yaml").read_bytes()
    if explicit:
        with pytest.raises(ValueError, match=FLAGS[0]) as exc:
            bind_route_entry(agent, entry, "custom:relay", "model")
        assert agent.__dict__ == before
        assert SECRET not in str(exc.value)
    else:
        with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
            assert bind_route_entry(agent, entry, "custom:relay", "model")
        assert agent.api_mode == "chat_completions"
        assert agent.capabilities == {FLAGS[0]: False, "future_flag": True}
        agent.client.close()
    assert (tmp_path / "config.yaml").read_bytes() == saved


@pytest.mark.parametrize("flag", FLAGS)
def test_invalid_provider_trust_is_not_hidden_by_model_false(tmp_path, monkeypatch, flag):
    save(tmp_path, monkeypatch, {flag: True}, "openai-chat", {flag: False})
    with pytest.raises(ValueError, match=flag):
        resolve_runtime_provider(requested="custom:relay", target_model="model")


@pytest.mark.parametrize("flag", FLAGS)
def test_explicit_unnamed_fallback_rejects_invalid_trust_before_mutation(tmp_path, monkeypatch, flag):
    from agent.route_binding import bind_route_entry
    from tests.agent.test_primary_runtime_restore import _make_agent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _make_agent(model="primary-model")
    before = dict(agent.__dict__)
    with pytest.raises(ValueError, match=flag):
        bind_route_entry(agent, {"api_mode": "openai-chat", "capabilities": {flag: True},
                                "base_url": "https://relay.example/v1", "api_key": SECRET}, "custom", "model")
    assert agent.__dict__ == before


def test_migration_dispatcher_reports_safe_rejection_without_losing_legacy_trust(tmp_path, monkeypatch, caplog):
    from hermes_cli.config_migrations import run_migrations
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {"_config_version": 11, "custom_providers": [{
        "name": "relay", "base_url": "https://relay.example/v1", "api_key": SECRET,
        "api_mode": "responses", "capabilities": {FLAGS[0]: SECRET},
    }]})
    result = {"config_added": [], "warnings": []}
    run_migrations(11, result, quiet=True)
    assert any(FLAGS[0] in warning for warning in result["warnings"])
    assert SECRET not in str(result) + caplog.text
    from hermes_cli.config import read_raw_config
    assert read_raw_config()["custom_providers"][0]["capabilities"] == {FLAGS[0]: SECRET}


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [SECRET, 0, 1, None, []])
@pytest.mark.parametrize("legacy", [False, True])
def test_flat_model_malformed_types_are_rejected(tmp_path, monkeypatch, flag, value, legacy):
    entry = save(tmp_path, monkeypatch, {}, "responses" if flag == FLAGS[0] else "messages", {flag: value})
    if legacy:
        atomic_config_replace(tmp_path / "config.yaml", {"custom_providers": [{
            **entry, "name": "relay", "base_url": entry["api"], "api_mode": entry["transport"],
        }]})
    with pytest.raises(ValueError, match=flag):
        resolve_runtime_provider(requested="custom:relay", target_model="model")


@pytest.mark.parametrize("flag", FLAGS)
def test_inferred_chat_wire_rejects_enabled_trust(tmp_path, monkeypatch, flag):
    save(tmp_path, monkeypatch, {flag: True}, "")
    with pytest.raises(ValueError, match=flag):
        resolve_runtime_provider(requested="custom:relay")


@pytest.mark.parametrize("flag", FLAGS)
def test_inherited_trust_deactivates_only_on_explicit_incompatible_wire(tmp_path, monkeypatch, flag):
    from agent.route_binding import bind_route_entry
    from tests.agent.test_primary_runtime_restore import _make_agent
    from unittest.mock import patch
    save(tmp_path, monkeypatch, {flag: True, "arbitrary_future": True},
         "responses" if flag == FLAGS[0] else "messages")
    agent = _make_agent(model="primary-model")
    with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
        assert bind_route_entry(agent, {"transport": "openai-chat"}, "custom:relay", "model")
    assert agent.api_mode == "chat_completions"
    assert agent.capabilities == {flag: False, "arbitrary_future": True}
    agent.client.close()


def test_direct_alias_pool_without_capabilities_remains_supported(monkeypatch):
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_custom import _resolve_direct_alias_runtime
    monkeypatch.setattr(rp, "_get_model_config", dict)
    pooled = {"provider": "custom", "api_mode": "chat_completions", "base_url": "https://relay.example/v1", "api_key": SECRET}
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: dict(pooled))
    result = _resolve_direct_alias_runtime("custom", None, "https://relay.example/v1")
    assert result["source"] == "direct-alias"
    assert "capabilities" not in result


@pytest.mark.parametrize("target_model", [None, "grok-4.5"])
def test_opencode_trust_uses_effective_global_or_target_model(tmp_path, monkeypatch, target_model):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {
        "model": {"default": "grok-4.5" if target_model is None else "claude-sonnet-4"},
        "providers": {"relay": {"api": "https://opencode.ai/zen/v1", "api_key": SECRET,
                                "capabilities": {FLAGS[0]: True}}},
    })
    runtime = resolve_runtime_provider(requested="custom:relay", target_model=target_model)
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["capabilities"][FLAGS[0]] is True


@pytest.mark.parametrize("flag,wire", [
    (FLAGS[0], "CODEX_RESPONSES"), (FLAGS[0], "RESPONSES"),
    (FLAGS[1], "ANTHROPIC_MESSAGES"), (FLAGS[1], "MESSAGES"),
    (FLAGS[2], "ANTHROPIC_MESSAGES"), (FLAGS[2], "ANTHROPIC-MESSAGES"),
])
def test_runtime_uppercase_canonical_and_alias_trust(tmp_path, monkeypatch, flag, wire):
    from hermes_cli.runtime_provider import _parse_api_mode
    expected = "codex_responses" if flag == FLAGS[0] else "anthropic_messages"
    assert _parse_api_mode(wire) == expected
    save(tmp_path, monkeypatch, {flag: True}, wire)
    runtime = resolve_runtime_provider(requested="custom:relay")
    assert runtime["api_mode"] == expected
    assert runtime["capabilities"][flag] is True


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("overlay", [False, True])
@pytest.mark.parametrize("enabled", [False, True, None])
def test_final_overlay_validates_effective_trust(tmp_path, monkeypatch, caplog, flag, overlay, enabled):
    from hermes_cli.config import read_raw_config
    wire = "codex_responses" if flag == FLAGS[0] else "anthropic_messages"
    caps = {"future.unregistered-capability": True}
    if enabled is not None:
        caps[flag] = enabled
    save(tmp_path, monkeypatch, caps, wire)
    config = read_raw_config()
    config["model"] = {"openai_runtime": "codex_app_server" if overlay else "default"}
    atomic_config_replace(tmp_path / "config.yaml", config)
    if overlay and enabled:
        with pytest.raises(ValueError) as exc:
            resolve_runtime_provider(requested="custom:relay")
        assert str(exc.value) == f"Capability {flag} requires {wire}"
        assert SECRET not in str(exc.value) + caplog.text
    else:
        runtime = resolve_runtime_provider(requested="custom:relay")
        assert runtime["api_mode"] == ("codex_app_server" if overlay else wire)
        assert runtime["capabilities"] == caps


@pytest.mark.parametrize("caps", [None, [], SECRET, True, 1])
def test_non_mapping_capability_container_retains_upstream_empty_map(tmp_path, monkeypatch, caps):
    save(tmp_path, monkeypatch, caps)
    runtime = resolve_runtime_provider(requested="custom:relay")
    assert runtime.get("capabilities", {}) == {}


def test_empty_container_rule_is_not_a_general_name_allowlist(tmp_path, monkeypatch):
    caps = {"future.unregistered-capability": True, "another_future_flag": False,
            "description": "generic non-boolean metadata"}
    save(tmp_path, monkeypatch, caps)
    runtime = resolve_runtime_provider(requested="custom:relay")
    assert runtime["capabilities"] == {"future.unregistered-capability": True, "another_future_flag": False}


@pytest.mark.parametrize("flag", FLAGS)
def test_unresolved_opencode_wire_still_rejects_raw_known_types(monkeypatch, flag):
    from hermes_cli.config_providers import get_compatible_custom_providers
    from hermes_cli import runtime_provider as rp
    def no_live_selection():
        raise AssertionError("Pure normalization must not select a live model")
    monkeypatch.setattr(rp, "_get_model_config", no_live_selection)
    config = {"providers": {"relay": {"api": "https://opencode.ai/zen/v1",
                                      "capabilities": {flag: SECRET}}}}
    with pytest.raises(ValueError, match=flag) as exc:
        get_compatible_custom_providers(config)
    assert SECRET not in str(exc.value)


@pytest.mark.parametrize("explicit_key", [False, True])
@pytest.mark.parametrize("model", ["grok-4.5", "claude-sonnet-4"])
def test_opencode_final_global_wire_is_checked_on_pool_and_direct_paths(tmp_path, monkeypatch, explicit_key, model):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {
        "model": {"default": model},
        "providers": {"relay": {"api": "https://opencode.ai/zen/v1", "api_key": SECRET,
                                "capabilities": {FLAGS[0]: True}}},
    })
    if model == "grok-4.5":
        runtime = resolve_runtime_provider(requested="custom:relay", explicit_api_key=SECRET if explicit_key else None)
        assert runtime["api_mode"] == "codex_responses"
        assert runtime["capabilities"][FLAGS[0]] is True
    else:
        with pytest.raises(ValueError, match=FLAGS[0]):
            resolve_runtime_provider(requested="custom:relay", explicit_api_key=SECRET if explicit_key else None)
