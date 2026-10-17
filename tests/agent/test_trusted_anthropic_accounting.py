"""Wire-preserved ordered thinking must count in real compression pressure."""
from copy import deepcopy
import logging

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.model_metadata import estimate_tokens_rough
from agent.turn_context import _preflight_request_tokens
from agent.turn_request_assembly import assemble_api_request
from tests.agent.test_anthropic_preserved_thinking_replay import (
    _agent, _carrier, _session_db, _patch_assembly_loop, _thinking_blocks,
)
from tests.agent.test_trusted_anthropic_capabilities import RELAY, MODEL


@pytest.mark.parametrize("flag,url,expected", [
    (True, RELAY, True), (False, RELAY, False), ("true", RELAY, False),
    (None, "https://api.anthropic.com", True),
])
def test_ordered_wire_preflight_assembly_tail_agree(tmp_path, monkeypatch, flag, url, expected):
    agent = _agent(_session_db(tmp_path, "s1"), model=MODEL, route=("custom", url))
    agent.capabilities = {"anthropic_signed_thinking": flag}
    agent.context_compressor.capabilities = agent.capabilities.copy()
    _patch_assembly_loop(monkeypatch)

    def measure(size):
        canonical = _carrier(thinking="x" * size) + [{"role": "user", "content": "continue"}]
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            canonical[1].pop(key)
        before = deepcopy(canonical)
        _, wire = convert_messages_to_anthropic(
            canonical, base_url=url, model=MODEL, preserve_signed_thinking=flag,
        )
        preflight = _preflight_request_tokens(agent, canonical, "")
        assembled = assemble_api_request(
            agent, messages=canonical, current_turn_user_idx=len(canonical)-1,
            _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
            active_system_prompt="", original_user_message="continue",
            pending_moa_prepared_request=None, request_logger=logging.getLogger(__name__),
        )
        tail = agent.context_compressor._walk_tail_budget(canonical, 0, 10**9, 0, cut_at_break=False)[1]
        assert canonical == before
        return (sum(estimate_tokens_rough(b.get("thinking", "")) for b in _thinking_blocks(wire)),
                preflight, assembled.approx_tokens, assembled.request_pressure_tokens, tail)

    small, large = measure(1), measure(8000)
    delta = tuple(b-a for a, b in zip(small, large))
    growth = estimate_tokens_rough("x" * 8000) - estimate_tokens_rough("x") if expected else 0
    if expected:
        assert delta == (growth,) * len(delta)
    else:
        assert delta[0] == 0
        assert agent.context_compressor._native_anthropic_budget() is None


def test_compressor_capability_lifecycle_init_switch_fallback_restore(tmp_path, monkeypatch):
    from unittest.mock import patch
    from hermes_cli.config import atomic_config_replace
    from tests.agent.test_primary_runtime_restore import _make_agent
    from agent.agent_runtime_helpers import _update_switch_compressor
    from agent.chat_completion_helpers import _update_fallback_context_compressor
    from agent.route_binding import reinstall_runtime_snapshot

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(tmp_path / "config.yaml", {})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200_000)
    agent = _make_agent(model=MODEL, base_url=RELAY,
                        capabilities={"anthropic_signed_thinking": True})
    cc = agent.context_compressor
    assert cc.capabilities == agent.capabilities
    assert cc.capabilities is not agent.capabilities
    agent.api_mode = "anthropic_messages"
    _update_switch_compressor(agent, [], 200_000, {})
    assert cc._native_anthropic_budget() is not None
    primary = dict(agent._primary_runtime)
    primary.update(api_mode=agent.api_mode, compressor_api_mode=agent.api_mode)
    primary["capabilities"] = dict(agent.capabilities)

    # Capability-only changes at the same endpoint must not reuse stale trust.
    agent.capabilities = {}
    _update_switch_compressor(agent, [], 200_000, {})
    assert cc._native_anthropic_budget() is None
    agent.capabilities = {"anthropic_signed_thinking": True}
    _update_fallback_context_compressor(agent)
    assert cc._native_anthropic_budget() is not None
    agent.capabilities = {}
    agent.base_url = "https://other.example/anthropic"
    _update_fallback_context_compressor(agent)
    assert cc._native_anthropic_budget() is None
    with patch("agent.agent_runtime_helpers._rebuild_primary_client"):
        reinstall_runtime_snapshot(agent, primary)
    assert cc.capabilities == {"anthropic_signed_thinking": True}
    assert cc._native_anthropic_budget() is not None


@pytest.fixture(autouse=True)
def offline_candidate_imports(monkeypatch):
    from pathlib import Path
    from agent import anthropic_thinking_replay, anthropic_thinking_policy, context_compressor

    root = Path(__file__).resolve().parents[2]
    for module in (anthropic_thinking_replay, anthropic_thinking_policy, context_compressor):
        assert Path(module.__file__).resolve().is_relative_to(root)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 200_000)
    def reject_network(*args, **kwargs):
        raise AssertionError("Signed-thinking contracts must not open network connections")
    monkeypatch.setattr("socket.socket.connect", reject_network)
