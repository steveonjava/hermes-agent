"""Native compaction opt-in through real named-provider resolution and agents."""

import copy
import json
from contextlib import closing
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent


MODELS = ("gpt-6.1-sol-chatgpt-tier", "gpt-6-luna-chatgpt-tier")
RELAY = "http://litellm.browsecode.org:4000"
DIRECTIVE = [{"type": "compaction", "compact_threshold": 220_000}]


@pytest.fixture
def route_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "model": {
            "provider": "custom:chatgpt-tier",
            "default": MODELS[0],
            "base_url": "http://litellm:4000",
            "context_length": 1_000_000,
        },
        "compression": {
            "enabled": True,
            "threshold": 0.5,
            "codex_responses_native": True,
            "codex_responses_compact_threshold": 220_000,
        },
        "custom_providers": [{
            "name": "chatgpt-tier",
            "base_url": RELAY,
            "api_key": "test-key",
            "api_mode": "codex_responses",
            "model": MODELS[0],
            "models": {model: {"context_length": 1_000_000} for model in MODELS},
            "capabilities": {"openai_native_compaction": True},
        }],
    }
    return config


def make_agent(config, model=MODELS[0]):
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump(config))

    runtime = resolve_runtime_provider(requested="custom:chatgpt-tier")
    assert runtime["base_url"] == RELAY
    return AIAgent(
        **{k: runtime[k] for k in (
            "provider", "requested_provider", "base_url", "api_key", "api_mode"
        )},
        capabilities=runtime.get("capabilities", {}),
        model=model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        enabled_toolsets=[],
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("format", ("legacy", "providers"))
def test_opted_in_named_route_sends_native_directive(route_config, model, format):
    if format == "providers":
        entry = route_config.pop("custom_providers")[0]
        route_config["providers"] = {"chatgpt-tier": entry}
    agent = make_agent(route_config, model)
    assert agent._build_api_kwargs([{"role": "user", "content": "hi"}])["context_management"] == DIRECTIVE
    assert agent.context_compressor.threshold_tokens == 500_000


@pytest.mark.parametrize("capability", (None, False, "true", 1, {}, []))
def test_capability_absent_or_malformed_fails_closed(route_config, capability):
    route_config["custom_providers"][0]["capabilities"] = {
        "openai_native_compaction": capability
    }
    agent = make_agent(route_config)
    assert "context_management" not in agent._build_api_kwargs([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("model", ("gpt-5.1", "gpt-5.2", "gpt-5.3-codex", "gpt-6-unknown", "claude-tier"))
def test_capability_does_not_enable_unsupported_models(route_config, model):
    agent = make_agent(route_config, model)
    assert "context_management" not in agent._build_api_kwargs([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("key", ("enabled", "codex_responses_native"))
def test_compression_kill_switch_wins_over_custom_extra_body(route_config, key):
    route_config["compression"][key] = False
    entry = route_config["custom_providers"][0]
    entry["extra_body"] = {"context_management": DIRECTIVE, "custom_field": "kept"}
    agent = make_agent(route_config)
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    assert "context_management" not in kwargs
    assert kwargs["extra_body"] == {"custom_field": "kept"}
    assert agent.request_overrides["extra_body"]["context_management"] == DIRECTIVE


def test_route_switch_does_not_carry_capability(route_config):
    agent = make_agent(route_config)
    agent.switch_model(new_model=agent.model, new_provider="custom:unapproved",
                       base_url="http://unapproved-relay:4000", api_key="test-key",
                       api_mode="codex_responses")
    assert "context_management" not in agent._build_api_kwargs([{"role": "user", "content": "hi"}])


def test_named_provider_switch_on_same_endpoint_does_not_carry_capability(route_config):
    agent = make_agent(route_config)
    agent.switch_model(new_model=agent.model, new_provider="custom:unapproved",
                       base_url=RELAY, api_key="test-key", api_mode="codex_responses")
    assert "context_management" not in agent._build_api_kwargs([{"role": "user", "content": "hi"}])


def test_migration_preserves_validated_capability(route_config):
    from hermes_cli.config import _custom_provider_entry_to_provider_config

    migrated = _custom_provider_entry_to_provider_config(route_config["custom_providers"][0])
    assert migrated["capabilities"] == {"openai_native_compaction": True}


def test_clamp_on_opted_in_route(route_config):
    agent = make_agent(route_config)
    agent.context_compressor.threshold_tokens = 100_000
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    assert kwargs["context_management"][0]["compact_threshold"] == 100_000 - 8192


def test_transport_cannot_override_native_directive():
    from agent.transports.codex import ResponsesApiTransport

    overrides = {"context_management": DIRECTIVE, "extra_body": {"context_management": DIRECTIVE}}
    original = copy.deepcopy(overrides)
    kwargs = ResponsesApiTransport().build_kwargs(
        model=MODELS[0], messages=[{"role": "user", "content": "hi"}],
        request_overrides=overrides, context_management=None,
    )
    assert "context_management" not in kwargs
    assert "context_management" not in kwargs.get("extra_body", {})
    assert overrides == original


def test_keyed_provider_opt_out_wins_over_legacy_opt_in(route_config):
    keyed = copy.deepcopy(route_config["custom_providers"][0])
    keyed["capabilities"]["openai_native_compaction"] = False
    route_config["providers"] = {"chatgpt-tier": keyed}
    agent = make_agent(route_config)
    assert "context_management" not in agent._build_api_kwargs([{"role": "user", "content": "hi"}])


def test_legacy_display_name_uses_same_aliases_as_resolver(route_config):
    route_config["custom_providers"][0]["name"] = "ChatGPT Tier"
    agent = make_agent(route_config)
    assert agent._build_api_kwargs([{"role": "user", "content": "hi"}])["context_management"] == DIRECTIVE


def test_route_api_mode_case_normalization_matches_resolver(route_config):
    route_config["custom_providers"][0]["api_mode"] = " CODEX_RESPONSES "
    agent = make_agent(route_config)
    assert agent._build_api_kwargs([{"role": "user", "content": "hi"}])["context_management"] == DIRECTIVE


@pytest.mark.parametrize("model", MODELS)
def test_real_sdk_wire_contains_only_gated_directive(route_config, model):
    import httpx
    from openai import OpenAI

    agent = make_agent(route_config, model)
    captured = []

    def server(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "resp_test", "status": "completed", "output": []})

    with OpenAI(
        api_key="test-key", base_url=RELAY + "/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    ) as client:
        client.responses.create(**agent._build_api_kwargs([{"role": "user", "content": "hi"}]))
        agent.codex_responses_native_compaction = False
        client.responses.create(**agent._build_api_kwargs([{"role": "user", "content": "hi"}]))
    assert captured[0]["context_management"] == DIRECTIVE
    assert "context_management" not in captured[1]


@pytest.mark.parametrize("rejected_parameter", ("context_management", "compact_threshold"))
def test_rejection_retries_once_and_remains_off(route_config, monkeypatch, rejected_parameter):
    import httpx
    from openai import BadRequestError

    route_config["custom_providers"][0]["extra_body"] = {"context_management": DIRECTIVE}
    agent = make_agent(route_config)
    agent._disable_streaming = True
    captured = []

    def request(kwargs):
        captured.append(copy.deepcopy(kwargs))
        if len(captured) == 1:
            response = httpx.Response(400, request=httpx.Request("POST", RELAY + "/v1/responses"))
            raise BadRequestError("Unknown parameter: " + rejected_parameter, response=response, body=None)
        return SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="DONE")])],
            status="completed", model=agent.model, usage=None,
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", request)
    result = agent.run_conversation("Reply DONE", system_message="Test session.")
    assert result["completed"] is True
    assert len(captured) == 2
    assert captured[0]["context_management"] == DIRECTIVE
    assert "context_management" not in captured[1]
    assert "context_management" not in captured[1].get("extra_body", {})
    assert agent.codex_responses_native_compaction is False
    assert agent.compression_enabled is True
    assert agent.context_compressor.should_compress(prompt_tokens=500_001)
    followup = agent.run_conversation("Reply DONE again", system_message="Test session.", conversation_history=result["messages"])
    assert followup["completed"] is True
    assert len(captured) == 3
    assert "context_management" not in captured[2]


def test_checkpoint_persistence_endpoint_binding_and_kill_switch(route_config, tmp_path):
    from agent.codex_responses_adapter import _normalize_codex_response
    from hermes_state import SessionDB

    agent = make_agent(route_config)
    agent._build_api_kwargs([{"role": "user", "content": "seed"}])
    issuer = agent._get_transport()._last_issuer_kind
    raw = SimpleNamespace(status="completed", output=[
        SimpleNamespace(type="compaction", encrypted_content="test-opaque-checkpoint"),
        SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="ACK")]),
    ])
    response, _ = _normalize_codex_response(raw, issuer_kind=issuer)
    with closing(SessionDB(db_path=tmp_path / "checkpoint.db")) as db:
        db.create_session("disposable", source="cli")
        db.append_message("disposable", "user", content="seed")
        db.append_message("disposable", "assistant", content="ACK", codex_reasoning_items=response.codex_reasoning_items)
    with closing(SessionDB(db_path=tmp_path / "checkpoint.db")) as db:
        history = db.get_messages_as_conversation("disposable")
    original = copy.deepcopy(history)
    same = agent._build_api_kwargs(history)
    assert any(i.get("type") == "compaction" for i in same["input"])
    assert "_issuer_kind" not in json.dumps(same["input"])
    agent.base_url = "http://foreign-issuer:4000"
    foreign = agent._build_api_kwargs(history)
    assert not any(i.get("type") == "compaction" for i in foreign["input"])
    assert history == original
    agent.base_url = RELAY
    agent._disable_codex_reasoning_replay(history)
    disabled = agent._build_api_kwargs(history)
    assert not any(i.get("type") == "compaction" for i in disabled["input"])


def test_anthropic_fallback_wire_has_no_encrypted_checkpoint(route_config):
    agent = make_agent(route_config)
    agent._build_api_kwargs([{"role": "user", "content": "seed"}])
    history = [
        {"role": "user", "content": "seed"},
        {"role": "assistant", "content": "ACK", "codex_reasoning_items": [{
            "type": "compaction", "encrypted_content": "test-opaque-checkpoint",
            "_issuer_kind": agent._get_transport()._last_issuer_kind,
        }]},
        {"role": "user", "content": "continue"},
    ]
    original = copy.deepcopy(history)
    agent.switch_model(
        new_model="hermes-anthropic-sonnet-tier", new_provider="custom:claude-tier",
        base_url=RELAY, api_key="test-key", api_mode="anthropic_messages",
    )
    kwargs = agent._build_api_kwargs(history)
    assert "context_management" not in kwargs
    assert "test-opaque-checkpoint" not in json.dumps(kwargs)
    assert history == original


def test_inherited_delegate_uses_native_route_without_new_routing(route_config):
    from tools.delegate_tool import _build_child_agent

    parent = make_agent(route_config)
    child = _build_child_agent(
        task_index=0, goal="Reply DONE", context=None, toolsets=None,
        model=None, max_iterations=2, task_count=1, parent_agent=parent,
    )
    assert child.model == parent.model
    assert child.base_url == parent.base_url
    assert child.api_mode == "codex_responses"
    assert child._build_api_kwargs([{"role": "user", "content": "hi"}])["context_management"] == DIRECTIVE


def test_auxiliary_inherits_subscription_runtime(route_config):
    from agent.auxiliary_client import CodexAuxiliaryClient, _resolve_auto_route

    parent = make_agent(route_config)
    client, model, _provider = _resolve_auto_route(main_runtime=parent._current_main_runtime(), task="compression")
    assert model == parent.model
    assert isinstance(client, CodexAuxiliaryClient)
    assert str(client.base_url).rstrip("/") == RELAY

