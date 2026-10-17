"""Trusted Messages capabilities must affect the wire without rewriting history."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import agent_runtime_helpers, anthropic_adapter, chat_completion_helpers
from agent.prompt_caching import build_prompt_cache_plan
from agent.transports.anthropic import AnthropicTransport


RELAY = "https://relay.example.com/anthropic"
MODEL = "claude-opus-4-8"


def test_real_imports_are_from_this_checkout():
    root = Path(__file__).resolve().parents[2]
    for module in (agent_runtime_helpers, anthropic_adapter, chat_completion_helpers):
        assert Path(module.__file__).resolve().is_relative_to(root)


@pytest.mark.parametrize(
    "flag,wire,url,expected",
    [
        (True, "anthropic_messages", RELAY, True),
        (None, "anthropic_messages", RELAY, False),
        (False, "anthropic_messages", RELAY, False),
        ("true", "anthropic_messages", RELAY, False),
        (1, "anthropic_messages", RELAY, False),
        (True, "chat_completions", RELAY, False),
        (True, "codex_responses", RELAY, False),
        (None, "anthropic_messages", "https://api.anthropic.com", True),
        (True, "chat_completions", "https://api.anthropic.com", False),
    ],
)
def test_tool_marker_capability_reaches_wire(flag, wire, url, expected):
    agent = SimpleNamespace(base_url=url, api_mode=wire,
                            capabilities={"anthropic_tool_cache_marker": flag})
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {
        "name": "example", "description": "Example", "parameters": {"type": "object"},
    }}]
    original = deepcopy((messages, tools, agent.capabilities))
    gate = agent_runtime_helpers._direct_native_anthropic_tool_cache_capability(agent)
    plan = build_prompt_cache_plan(messages, tools, native_anthropic=True,
                                   direct_native_tool_cache=gate)
    kwargs = AnthropicTransport().build_kwargs(model=MODEL, messages=plan.messages,
                                               tools=plan.tools, base_url=url)
    assert ("cache_control" in kwargs["tools"][-1]) is expected
    assert (messages, tools, agent.capabilities) == original


def _signed_history():
    return [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "answer", "reasoning_details": [
            {"type": "thinking", "thinking": "signed plan", "signature": "synthetic-signature",
             "cache_control": {"type": "ephemeral"}},
            {"type": "thinking", "thinking": "unsigned plan"},
            {"type": "redacted_thinking", "data": "synthetic-redacted"},
            {"type": "redacted_thinking"},
        ]},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "latest"},
    ]


def _request_agent(flag, url):
    return SimpleNamespace(
        model=MODEL, provider="custom", capabilities={"anthropic_signed_thinking": flag},
        max_tokens=4096, _is_anthropic_oauth=False, _anthropic_base_url=url,
        _get_transport=AnthropicTransport, _anthropic_preserve_dots=lambda: False,
        _prepare_anthropic_messages_for_api=lambda messages: messages,
    )


@pytest.mark.parametrize("flag,url,expected", [
    (True, RELAY, True),
    (None, RELAY, False),
    (False, RELAY, False),
    ("true", RELAY, False),
    (1, RELAY, False),
    (None, "https://api.anthropic.com", True),
    (None, "https://inference-api.nousresearch.com/v1", True),
    (None, "https://aiplatform.googleapis.com", True),
])
def test_signed_thinking_opt_in_survives_request_path(flag, url, expected):
    messages = _signed_history()
    original = deepcopy(messages)
    kwargs = chat_completion_helpers._build_anthropic_kwargs(
        _request_agent(flag, url), messages, None, None, {},
    )
    blocks = kwargs["messages"][1]["content"]
    thinking = [b for b in blocks if b["type"] in {"thinking", "redacted_thinking"}]
    assert thinking == ([
        {"type": "thinking", "thinking": "signed plan", "signature": "synthetic-signature"},
        {"type": "redacted_thinking", "data": "synthetic-redacted"},
    ] if expected else [])
    if expected:
        assert {"type": "text", "text": "unsigned plan"} in blocks
    assert "preserve_signed_thinking" not in kwargs
    assert messages == original


@pytest.mark.parametrize("flag,expected", [(True, True), (False, False), ("true", False), (1, False)])
def test_transport_conversion_honors_only_boolean_trust(flag, expected):
    messages = _signed_history()
    original = deepcopy(messages)
    _, converted = AnthropicTransport().convert_messages(
        messages, base_url=RELAY, model=MODEL, preserve_signed_thinking=flag,
    )
    thinking = [b for b in converted[1]["content"] if b["type"] == "thinking"]
    assert bool(thinking) is expected
    assert messages == original


@pytest.mark.parametrize("url,model", [
    ("https://api.kimi.com/coding", "kimi-k2.5"),
    ("https://api.deepseek.com/anthropic", "deepseek-v4-pro"),
    ("https://api.minimax.io/anthropic", "MiniMax-M2.7"),
])
def test_trust_does_not_replace_known_platform_replay_contract(url, model):
    transport = AnthropicTransport()
    messages = _signed_history()
    original = deepcopy(messages)
    baseline = transport.convert_messages(messages, base_url=url, model=model)
    trusted = transport.convert_messages(messages, base_url=url, model=model, preserve_signed_thinking=True)
    assert trusted == baseline
    assert messages == original
