"""Offline constructor snapshot, migration, and signed-carrier lifecycle contracts."""
from copy import deepcopy
import json
from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import _build_anthropic_kwargs
from agent.model_metadata import estimate_tokens_rough
from agent.turn_context import _preflight_request_tokens
from tests.agent.test_anthropic_preserved_thinking_replay import _carrier, _thinking_blocks
from tests.agent.test_retained_switch_capability_contracts import switch_agent
from tests.agent.test_trusted_anthropic_capabilities import MODEL, RELAY


@pytest.mark.parametrize("lifecycle", ["initial", "constructor_restore", "public_switch_restore"])
def test_real_constructor_fallback_restore_preserves_accounting(switch_agent, monkeypatch, lifecycle):
    flags = {"anthropic_signed_thinking": True}
    agent = switch_agent(flags)
    if lifecycle == "public_switch_restore":
        agent.switch_model(MODEL, "custom", base_url=RELAY,
                           api_mode="anthropic_messages", provider_capabilities=flags)
    primary = deepcopy(agent._primary_runtime)
    if lifecycle != "initial":
        client = MagicMock(base_url="https://fallback.example/anthropic")
        monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                            lambda *a, **k: (client, MODEL))
        agent._fallback_chain = [{"provider": "custom", "model": MODEL,
                                  "base_url": str(client.base_url), "api_key": "synthetic-fallback-key",
                                  "api_mode": "anthropic_messages", "capabilities": {}}]
        agent._fallback_index = 0
        assert agent._try_activate_fallback()
        assert agent._primary_runtime == primary
        assert agent._restore_primary_runtime()
        assert agent._primary_runtime == primary
    def measure(size):
        history = _carrier(thinking="x" * size) + [{"role": "user", "content": "continue"}]
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            history[1].pop(key)
        unchanged = deepcopy(history)
        kwargs = _build_anthropic_kwargs(agent, history, None, None, {})
        wire = sum(estimate_tokens_rough(b.get("thinking", ""))
                   for b in _thinking_blocks(kwargs["messages"]))
        preflight = _preflight_request_tokens(agent, history, "")
        tail = agent.context_compressor._walk_tail_budget(
            history, 0, 10**9, 0, cut_at_break=False)[1]
        assert history == unchanged
        return wire, preflight, tail
    small, large = measure(1), measure(8000)
    delta = tuple(b-a for a, b in zip(small, large))
    growth = estimate_tokens_rough("x" * 8000) - estimate_tokens_rough("x")
    print(json.dumps({"lifecycle": lifecycle, "agent_wire": agent.api_mode,
                      "compressor_wire": agent.context_compressor.api_mode,
                      "snapshot_has_compressor_wire": "compressor_api_mode" in primary,
                      "wire_preflight_tail_delta": delta, "expected_growth": growth}))
    assert delta == (growth,) * 3
    assert agent.context_compressor.api_mode == agent.api_mode == "anthropic_messages"


@pytest.fixture
def old_schema_agent(switch_agent):
    """Intentional migration fixture, NOT the untouched constructor end-to-end path."""
    agent = switch_agent({"anthropic_signed_thinking": True})
    agent._primary_runtime.pop("compressor_api_mode", None)
    return agent


@pytest.mark.parametrize("saved_wire", [None, "chat_completions", ""])
def test_restore_old_schema_uses_saved_effective_wire_only_when_missing(
    old_schema_agent, monkeypatch, saved_wire,
):
    agent = old_schema_agent
    if saved_wire is not None:
        agent._primary_runtime["compressor_api_mode"] = saved_wire
    primary = deepcopy(agent._primary_runtime)
    client = MagicMock(base_url="https://fallback.example/anthropic")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                        lambda *a, **k: (client, MODEL))
    agent._fallback_chain = [{"provider": "custom", "model": MODEL,
                              "base_url": str(client.base_url), "api_key": "synthetic-fallback-key",
                              "api_mode": "anthropic_messages", "capabilities": {}}]
    agent._fallback_index = 0
    assert agent._try_activate_fallback()
    assert agent._restore_primary_runtime()
    assert agent._primary_runtime == primary
    expected = primary["api_mode"] if saved_wire is None else saved_wire
    assert agent.context_compressor.api_mode == expected
    assert (agent.context_compressor._native_anthropic_budget() is not None) is (saved_wire is None)


def test_constructor_fallback_restore_keeps_rejection_suppression(
    switch_agent, monkeypatch, tmp_path,
):
    import logging
    from types import SimpleNamespace
    from agent.error_classifier import FailoverReason
    from agent.turn_recovery import _recover_format_errors
    from agent.turn_retry_state import TurnRetryState
    from agent.turn_request_assembly import assemble_api_request
    from tests.agent.test_anthropic_preserved_thinking_replay import _session_db, _signed_turn

    agent = switch_agent({"anthropic_signed_thinking": True})
    primary = deepcopy(agent._primary_runtime)  # Never amend this constructor snapshot.
    db = _session_db(tmp_path, "signed-lifecycle")
    agent._session_db, agent.session_id = db, "signed-lifecycle"
    agent._persist_disabled = False
    agent.context_compressor.bind_session_state(db, agent.session_id)
    from agent.turn_context import _reset_per_turn_agent_state
    _reset_per_turn_agent_state(agent)
    canonical = _carrier(thinking="rejected" * 1000) + [{"role": "user", "content": "continue"}]
    before = deepcopy(canonical)
    request = deepcopy(canonical)
    assert _recover_format_errors(
        agent, RuntimeError("Invalid signature"),
        SimpleNamespace(reason=FailoverReason.thinking_signature), TurnRetryState(), canonical, request,
    )
    assert canonical == before
    assert db.get_session_model_config_value(agent.session_id, "_anthropic_rejected_thinking", [])

    def measure(size):
        new_turn = _signed_turn("new question", "new answer", "sig-new", thinking="x" * size)
        new_turn[1]["anthropic_content_blocks"] = deepcopy(new_turn[1]["reasoning_details"])
        history = deepcopy(canonical[:-1]) + new_turn + deepcopy(canonical[-1:])
        for row in history:
            for key in ("reasoning", "reasoning_content", "reasoning_details"):
                row.pop(key, None)
        unchanged = deepcopy(history)
        assembled = assemble_api_request(
            agent, messages=history, current_turn_user_idx=len(history) - 1,
            _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
            active_system_prompt="", original_user_message="continue",
            pending_moa_prepared_request=None, request_logger=logging.getLogger(__name__),
        )
        kwargs = _build_anthropic_kwargs(agent, assembled.api_messages, None, None, {})
        blocks = _thinking_blocks(kwargs["messages"])
        trusted = bool(agent.capabilities.get("anthropic_signed_thinking"))
        assert [b.get("signature") for b in blocks if b.get("type") == "thinking"] == (
            ["sig-new"] if trusted else [])
        values = (sum(estimate_tokens_rough(b.get("thinking", "")) for b in blocks),
                  _preflight_request_tokens(agent, history, ""),
                  agent.context_compressor._walk_tail_budget(history, 0, 10**9, 0, cut_at_break=False)[1])
        assert history == unchanged
        return values

    client = MagicMock(base_url="https://fallback.example/anthropic")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                        lambda *a, **k: (client, MODEL))
    agent._fallback_chain = [{"provider": "custom", "model": MODEL,
                              "base_url": str(client.base_url), "api_key": "synthetic-fallback-key",
                              "api_mode": "anthropic_messages", "capabilities": {}}]
    agent._fallback_index = 0
    for phase in ("constructor", "fallback", "restored"):
        if phase == "fallback":
            assert agent._try_activate_fallback()
        elif phase == "restored":
            assert agent._restore_primary_runtime()
        small, large = measure(1), measure(8000)
        delta = tuple(b - a for a, b in zip(small, large))
        if phase != "fallback":
            growth = estimate_tokens_rough("x" * 8000) - estimate_tokens_rough("x")
            assert delta == (growth,) * 3
            assert agent.context_compressor._native_anthropic_budget() is not None
        else:
            assert delta[0] == 0
            assert agent.context_compressor._native_anthropic_budget() is None
        assert agent._primary_runtime == primary
    assert canonical == before
    db.close()
