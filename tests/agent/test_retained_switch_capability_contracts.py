"""Offline contracts through the retained public live-switch API."""
from copy import deepcopy
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent
from tests.agent.test_trusted_anthropic_capabilities import MODEL, RELAY


@pytest.fixture
def switch_agent(tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_replace
    from agent import agent_runtime_helpers, context_compressor
    import run_agent

    root = Path(__file__).resolve().parents[2]
    for module in (run_agent, agent_runtime_helpers, context_compressor):
        assert Path(module.__file__).resolve().is_relative_to(root)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    def reject_network(*args, **kwargs):
        raise AssertionError("Live-switch tests must remain offline")
    monkeypatch.setattr("socket.socket.connect", reject_network)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", lambda *a, **k: MagicMock())
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200_000)
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **k: 200_000)

    def make(capabilities=None):
        return AIAgent(
            model=MODEL, provider="custom", base_url=RELAY, api_mode="anthropic_messages",
            api_key="synthetic-switch-key", capabilities=capabilities or {},
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
    return make


@pytest.mark.parametrize("trusted", [True, False])
@pytest.mark.parametrize("failure", ["client", "compressor"])
def test_failed_public_switch_keeps_old_trust(switch_agent, monkeypatch, trusted, failure):
    flags = {"anthropic_signed_thinking": trusted, "anthropic_tool_cache_marker": trusted}
    agent = switch_agent(flags)
    primary = agent._primary_runtime
    old_client = agent._anthropic_client
    # An earlier config policy can differ from the policy re-read during the switch.
    agent._use_prompt_caching = agent._use_native_cache_layout = False
    old_cache = (agent._use_prompt_caching, agent._use_native_cache_layout)
    old_runtime = dict(agent.runtime_capabilities)
    old_cc = dict(agent.context_compressor.capabilities)
    def fail(*args, **kwargs):
        raise RuntimeError("forced rebuild failure")
    if failure == "client":
        monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", fail)
    else:
        monkeypatch.setattr(agent.context_compressor, "update_model", fail)
    with pytest.raises(RuntimeError, match="forced rebuild failure"):
        agent.switch_model(MODEL, "custom", base_url=RELAY, api_mode="anthropic_messages",
                           provider_capabilities={key: not value for key, value in flags.items()},
                           runtime_capabilities={"synthetic_runtime": True})
    assert agent.capabilities == flags
    assert agent.runtime_capabilities == old_runtime
    assert agent.context_compressor.capabilities == old_cc
    assert (agent._use_prompt_caching, agent._use_native_cache_layout) == old_cache
    assert agent._primary_runtime is primary
    assert agent._anthropic_client is old_client


@pytest.mark.parametrize("flag,wire,model", [
    ("anthropic_tool_cache_marker", "anthropic_messages", MODEL),
    ("anthropic_signed_thinking", "anthropic_messages", MODEL),
    ("openai_native_compaction", "codex_responses", "gpt-5.6"),
])
def test_legacy_provider_trust_flags_reach_public_switch(switch_agent, flag, wire, model):
    agent = switch_agent()
    agent.switch_model(model, "custom", base_url=RELAY, api_mode=wire,
                       capabilities={flag: True, "synthetic_runtime": True})
    assert agent.capabilities == {flag: True}
    assert agent.runtime_capabilities["synthetic_runtime"] is True
    assert agent._primary_runtime["capabilities"] == {flag: True}
    if flag == "openai_native_compaction":
        assert agent.runtime_capabilities["native_compaction"] is True
    elif flag == "anthropic_signed_thinking":
        assert agent.context_compressor._native_anthropic_budget() is not None
    else:
        assert agent._direct_native_anthropic_tool_cache_capability()


from agent.model_metadata import estimate_tokens_rough
from agent.turn_context import _preflight_request_tokens
from tests.agent.test_anthropic_preserved_thinking_replay import _carrier, _thinking_blocks


@pytest.mark.parametrize("trusted", [True, False])
@pytest.mark.parametrize("same_endpoint", [True, False])
def test_public_switch_propagates_destination_trust(switch_agent, monkeypatch, trusted, same_endpoint):
    from agent.chat_completion_helpers import _build_anthropic_kwargs
    from agent.prompt_caching import build_prompt_cache_plan
    flags = {"anthropic_signed_thinking": trusted, "anthropic_tool_cache_marker": trusted}
    old = {key: not value for key, value in flags.items()}
    agent = switch_agent(old)
    url = RELAY if same_endpoint else "https://other.example/anthropic"
    agent.switch_model(MODEL, "custom", base_url=url, api_mode="anthropic_messages",
                       provider_capabilities=flags)
    assert agent.capabilities == flags
    assert agent.context_compressor.capabilities == flags
    assert agent.context_compressor.capabilities is not agent.capabilities
    assert agent._primary_runtime["capabilities"] == flags
    assert agent._primary_runtime["runtime_capabilities"] == agent.runtime_capabilities
    assert (agent.context_compressor._native_anthropic_budget() is not None) is trusted
    tools = [{"type": "function", "function": {"name": "search",
              "description": "Synthetic tool", "parameters": {"type": "object"}}}]
    plan = build_prompt_cache_plan(
        [{"role": "user", "content": "hello"}], tools,
        native_anthropic=True, direct_native_tool_cache=agent._direct_native_anthropic_tool_cache_capability(),
    )
    from agent.transports.anthropic import AnthropicTransport
    cached_wire = AnthropicTransport().build_kwargs(model=MODEL, messages=plan.messages, tools=plan.tools, base_url=url)
    assert ("cache_control" in cached_wire["tools"][-1]) is trusted
    assert agent._direct_native_anthropic_tool_cache_capability() is trusted
    from agent.turn_request_assembly import assemble_api_request
    from tests.agent.test_anthropic_preserved_thinking_replay import _patch_assembly_loop
    _patch_assembly_loop(monkeypatch)
    agent._current_turn_timestamp = 1.0
    # Exercise real conversion, preflight and compressor accounting, not injected flags.
    def measure(size):
        canonical = _carrier(thinking="x" * size) + [{"role": "user", "content": "continue"}]
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            canonical[1].pop(key)
        before = deepcopy(canonical)
        kwargs = _build_anthropic_kwargs(agent, canonical, None, None, {})
        wire = sum(estimate_tokens_rough(b.get("thinking", "")) for b in _thinking_blocks(kwargs["messages"]))
        preflight = _preflight_request_tokens(agent, canonical, "")
        assembled = assemble_api_request(
            agent, messages=canonical, current_turn_user_idx=len(canonical)-1,
            _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
            active_system_prompt="", original_user_message="continue",
            pending_moa_prepared_request=None, request_logger=logging.getLogger(__name__),
        )
        tail = agent.context_compressor._walk_tail_budget(canonical, 0, 10**9, 0, cut_at_break=False)[1]
        assert canonical == before
        return wire, preflight, tail, assembled.approx_tokens, assembled.request_pressure_tokens
    small, large = measure(1), measure(8000)
    delta = tuple(b - a for a, b in zip(small, large))
    if trusted:
        growth = estimate_tokens_rough("x" * 8000) - estimate_tokens_rough("x")
        assert delta == (growth,) * 5
    else:
        assert delta[0] == 0
    primary = deepcopy(agent._primary_runtime)
    from unittest.mock import MagicMock
    fallback_client = MagicMock(base_url="https://fallback.example/anthropic")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: (fallback_client, MODEL))
    agent._fallback_chain = [{"provider": "custom", "model": MODEL,
                              "base_url": str(fallback_client.base_url), "api_key": "synthetic-fallback-key",
                              "api_mode": "anthropic_messages", "capabilities": {}}]
    agent._fallback_index = 0
    assert agent._try_activate_fallback()
    assert agent.capabilities == {}
    assert agent.context_compressor._native_anthropic_budget() is None
    assert agent._primary_runtime == primary
    assert agent._restore_primary_runtime()
    assert agent.capabilities == flags
    assert agent.context_compressor.capabilities == flags
    assert agent._direct_native_anthropic_tool_cache_capability() is trusted


@pytest.mark.parametrize("change", ["endpoint", "identity", "wire", "provider"])
@pytest.mark.parametrize("runtime", [{}, {"synthetic_runtime": True, "native_compaction": True}])
def test_legacy_runtime_map_does_not_carry_trust_to_another_route(switch_agent, change, runtime):
    agent = switch_agent({"anthropic_signed_thinking": True, "anthropic_tool_cache_marker": True})
    agent.requested_provider = "custom:old"
    kwargs = dict(base_url=RELAY, api_mode="anthropic_messages", requested_provider="custom:old")
    provider = "custom"
    if change == "endpoint":
        kwargs["base_url"] = "https://other.example/anthropic"
    elif change == "identity":
        kwargs["requested_provider"] = "custom:new"
    elif change == "wire":
        kwargs["api_mode"] = "chat_completions"
    else:
        provider = "openrouter"
    agent.switch_model(MODEL, provider, capabilities=runtime, **kwargs)
    assert agent.capabilities == {}
    assert agent.context_compressor.capabilities == {}
    assert agent.runtime_capabilities["native_compaction"] is False


@pytest.mark.parametrize("argument", ["capabilities", "provider_capabilities"])
@pytest.mark.parametrize("flag,wire", [
    ("anthropic_tool_cache_marker", "anthropic_messages"),
    ("anthropic_signed_thinking", "anthropic_messages"),
    ("openai_native_compaction", "codex_responses"),
])
@pytest.mark.parametrize("invalid", ["true", 1, None, "wrong-wire"])
def test_invalid_explicit_trust_is_rejected_before_mutation(switch_agent, monkeypatch, argument, flag, wire, invalid):
    agent = switch_agent({"anthropic_signed_thinking": True})
    primary = agent._primary_runtime
    old = (agent.model, agent.api_mode, agent.capabilities, agent.runtime_capabilities,
           agent._anthropic_client, agent._use_prompt_caching)
    def unexpected_builder(*a, **k):
        pytest.fail("Invalid opt-ins must fail before client rebuild")
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", unexpected_builder)
    value = True if invalid == "wrong-wire" else invalid
    if invalid == "wrong-wire":
        wire = "chat_completions"
    with pytest.raises(ValueError, match="Capability"):
        agent.switch_model("changed-model", "custom", base_url=RELAY, api_mode=wire,
                           **{argument: {flag: value}})
    assert agent._primary_runtime is primary
    assert (agent.model, agent.api_mode, agent.capabilities, agent.runtime_capabilities,
            agent._anthropic_client, agent._use_prompt_caching) == old


@pytest.mark.parametrize("runtime", [{}, {"synthetic_runtime": True, "native_compaction": True}])
def test_same_route_legacy_runtime_reselect_preserves_trust(switch_agent, runtime):
    flags = {"anthropic_signed_thinking": True}
    agent = switch_agent(flags)
    agent.switch_model(MODEL, "custom", api_mode="anthropic_messages", capabilities=runtime)
    assert agent.capabilities == flags
    assert agent.runtime_capabilities["native_compaction"] is False
    assert agent.context_compressor.capabilities == flags


@pytest.mark.parametrize("modern_runtime", [None, {"other_runtime": True, "native_compaction": True}])
def test_explicit_modern_arguments_win_without_generic_provider_grants(switch_agent, modern_runtime):
    agent = switch_agent({"anthropic_signed_thinking": True})
    kwargs = {"runtime_capabilities": modern_runtime} if modern_runtime is not None else {}
    agent.switch_model(MODEL, "custom", base_url=RELAY, api_mode="anthropic_messages",
                       capabilities={"anthropic_signed_thinking": "ignored", "synthetic_runtime": True},
                       provider_capabilities={}, **kwargs)
    assert agent.capabilities == {}
    assert agent.context_compressor.capabilities == {}
    assert agent.runtime_capabilities["native_compaction"] is False
    assert "synthetic_runtime" not in agent.runtime_capabilities
    if modern_runtime is not None:
        assert agent.runtime_capabilities["other_runtime"] is True


def test_partial_compressor_failure_cannot_leave_destination_trust(switch_agent, monkeypatch):
    agent = switch_agent({"anthropic_signed_thinking": True})
    old_capabilities = dict(agent.context_compressor.capabilities)
    primary = agent._primary_runtime
    update = agent.context_compressor.update_model
    def partial_failure(*a, **k):
        update(*a, **k)
        raise RuntimeError("partial compressor failure")
    monkeypatch.setattr(agent.context_compressor, "update_model", partial_failure)
    with pytest.raises(RuntimeError, match="partial compressor failure"):
        agent.switch_model(MODEL, "custom", base_url="https://different.example/anthropic",
                           api_mode="anthropic_messages", provider_capabilities={})
    assert agent.capabilities == old_capabilities
    assert agent.context_compressor.capabilities == old_capabilities
    assert agent._primary_runtime is primary
