from types import SimpleNamespace

from agent.native_compaction import (
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)


def _agent(capabilities):
    return SimpleNamespace(
        model="gpt-5.6",
        base_url="https://proxy.example/v1",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        codex_responses_compact_threshold=200_000,
        context_compressor=None,
        capabilities=capabilities,
    )


def test_trusted_destination_capabilities_are_explicitly_enabled():
    capabilities = resolve_native_compaction_capabilities(
        model="gpt-5.6",
        base_url="https://api.openai.com/v1",
    )

    assert capabilities == {"native_compaction": True}


def test_untrusted_destination_capabilities_are_explicitly_denied():
    capabilities = resolve_native_compaction_capabilities(
        model="gpt-5.6",
        base_url="https://openrouter.ai/api/v1",
    )

    assert capabilities == {"native_compaction": False}


def test_default_openai_destination_is_enabled_without_explicit_base_url():
    capabilities = resolve_native_compaction_capabilities(
        model="gpt-5.6",
        base_url="",
        provider="openai",
    )

    assert capabilities == {"native_compaction": True}


def test_trusted_proxy_production_capability_maps_emit_native_payload():
    provider_capabilities = {"openai_native_compaction": True}
    runtime_capabilities = resolve_native_compaction_capabilities(
        model="gpt-5.6-sol-chatgpt-tier",
        base_url="https://trusted-proxy.example/v1",
        provider="custom:trusted-proxy",
        provider_capabilities=provider_capabilities,
    )
    agent = _agent(provider_capabilities)
    agent.model = "gpt-5.6-sol-chatgpt-tier"
    agent.runtime_capabilities = runtime_capabilities

    assert native_compaction_context_management(
        agent, is_codex_backend=False
    ) == [{"type": "compaction", "compact_threshold": 200_000}]


def test_explicit_false_capability_denies_native_payload():
    agent = _agent({"native_compaction": False})

    assert native_compaction_context_management(agent, is_codex_backend=False) is None


def test_malformed_runtime_capability_denies_native_payload():
    agent = _agent({"openai_native_compaction": True})
    agent.runtime_capabilities = {"native_compaction": "true"}

    assert native_compaction_context_management(agent, is_codex_backend=False) is None


def test_missing_capability_keeps_default_deny():
    agent = _agent({})

    assert native_compaction_context_management(agent, is_codex_backend=False) is None


def test_aiagent_switch_model_forwards_separate_capability_maps(monkeypatch):
    from run_agent import AIAgent

    forwarded = {}

    def fake_switch_model(agent, *args, **kwargs):
        forwarded["agent"] = agent
        forwarded["args"] = args
        forwarded["kwargs"] = kwargs
        return "switched"

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch_model)
    agent = AIAgent.__new__(AIAgent)
    provider_capabilities = {"openai_native_compaction": True}
    runtime_capabilities = {"native_compaction": True}

    result = agent.switch_model(
        new_model="gpt-5.6-sol-chatgpt-tier",
        new_provider="custom:trusted-proxy",
        api_key="test-key",
        base_url="https://trusted-proxy.example/v1",
        api_mode="chat_completions",
        provider_capabilities=provider_capabilities,
        runtime_capabilities=runtime_capabilities,
    )

    assert result == "switched"
    assert forwarded == {
        "agent": agent,
        "args": (
            "gpt-5.6-sol-chatgpt-tier",
            "custom:trusted-proxy",
            "test-key",
            "https://trusted-proxy.example/v1",
            "chat_completions",
        ),
        "kwargs": {
            "capabilities": None,
            "provider_capabilities": provider_capabilities,
            "runtime_capabilities": runtime_capabilities,
        },
    }


def test_aiagent_switch_model_preserves_legacy_capabilities_argument(monkeypatch):
    from run_agent import AIAgent

    forwarded = {}

    def fake_switch_model(agent, *args, **kwargs):
        forwarded.update(kwargs)

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch_model)
    agent = AIAgent.__new__(AIAgent)
    legacy_capabilities = {"native_compaction": False}

    agent.switch_model(
        "legacy-model",
        "legacy-provider",
        "legacy-key",
        "https://legacy.example/v1",
        "chat_completions",
        legacy_capabilities,
    )

    assert forwarded == {
        "capabilities": legacy_capabilities,
        "provider_capabilities": None,
        "runtime_capabilities": None,
    }
