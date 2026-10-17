"""Trusted relay signature recovery must survive request rebuild and process resume."""
from copy import deepcopy

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState
from agent.error_classifier import FailoverReason
from types import SimpleNamespace
from tests.agent.test_anthropic_preserved_thinking_replay import (
    _agent, _carrier, _session_db, _patch_assembly_loop, _select_canonical_clone,
    _thinking_blocks, _signed_turn,
)
from tests.agent.test_trusted_anthropic_capabilities import RELAY, MODEL


@pytest.mark.parametrize("flag,url,durable", [
    (True, RELAY, True), (False, RELAY, False), ("true", RELAY, False),
    (True, "https://api.anthropic.com", True),
    (True, "https://api.minimax.io/anthropic", True),
    (True, "https://api.kimi.com/coding", False),
    (True, "https://api.deepseek.com/anthropic", False),
])
def test_ordered_signature_recovery_retry_rebuild_resume(tmp_path, monkeypatch, flag, url, durable):
    from agent.turn_request_assembly import assemble_api_request
    import logging

    db = _session_db(tmp_path, "s1")
    agent = _agent(db, model=MODEL, route=("custom", url))
    agent.capabilities = {"anthropic_signed_thinking": flag}
    canonical = _carrier() + [{"role": "user", "content": "continue"}]
    before = deepcopy(canonical)
    request = deepcopy(canonical)
    assert _recover_format_errors(
        agent, RuntimeError("Invalid signature"),
        SimpleNamespace(reason=FailoverReason.thinking_signature), TurnRetryState(), canonical, request,
    )
    convert = lambda rows: convert_messages_to_anthropic(
        rows, base_url=url, model=MODEL, preserve_signed_thinking=flag,
    )[1]
    if durable:
        assert not _thinking_blocks(convert(request))
        assert db.get_session_model_config_value("s1", "_anthropic_rejected_thinking", [])
    else:
        assert not db.get_session_model_config_value("s1", "_anthropic_rejected_thinking", [])

    assert canonical == before
    canonical = canonical[:-1] + _signed_turn("new", "new answer", "sig-new") + canonical[-1:]
    before = deepcopy(canonical)
    _patch_assembly_loop(monkeypatch, _select_canonical_clone)
    for reader in (agent, _agent(db, model=MODEL, route=("custom", url))):
        reader.capabilities = agent.capabilities.copy()
        assembled = assemble_api_request(
            reader, messages=canonical, current_turn_user_idx=len(canonical) - 1,
            _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
            active_system_prompt="", original_user_message="continue",
            pending_moa_prepared_request=None, request_logger=logging.getLogger(__name__),
        )
        if durable:
            wire = convert(assembled.api_messages)
            assert [b.get("signature") for b in _thinking_blocks(wire)] == ["sig-new"]
            assert "tool_use" in repr(wire) and "answer" in repr(wire)
    assert canonical == before


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
