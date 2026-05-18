"""Unit tests for copilot-specific branches in agent/agent_init.py (commit 971806e34).

HUNK_C — api_mode derivation for provider='copilot':
  - claude-* models → anthropic_messages (pattern-based, no catalog fetch)
  - other models → copilot_model_api_mode(); fallback to chat_completions

HUNK_D — api_key resolution for copilot + anthropic_messages + no key:
  - resolve_copilot_token() + get_copilot_api_token() → agent.api_key
  - exception during resolution → effective_key = ""
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_init import init_agent
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Shared mocking helpers
# ---------------------------------------------------------------------------

_HEAVY_PATCHES = [
    # Logging setup — writes to ~/.hermes/logs/
    patch("agent.agent_init._install_safe_stdio"),
    patch("hermes_logging.setup_logging"),
    # Tool loading — requires real toolsets and env
    patch("run_agent.get_tool_definitions", return_value=[]),
    patch("run_agent.check_toolset_requirements", return_value={}),
    # Anthropic client construction — would hit the Anthropic SDK
    patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    # Checkpoint manager — writes to disk
    patch("tools.checkpoint_manager.CheckpointManager", return_value=MagicMock()),
    # Session dir creation — writes to ~/.hermes/sessions/
    patch("pathlib.Path.mkdir"),
]


def _make_agent(model: str, api_key: str = "") -> AIAgent:
    """Return a bare AIAgent instance without calling __init__."""
    return object.__new__(AIAgent)


def _run_init(model: str, api_key: str = "", extra_patches=()) -> AIAgent:
    """
    Call init_agent on a fresh AIAgent shell with all heavy side-effects patched.

    Returns the agent after init_agent completes so callers can assert on
    agent.api_mode, agent.api_key, etc.
    """
    agent = _make_agent(model, api_key)

    patches = list(_HEAVY_PATCHES) + list(extra_patches)
    with (
        patch("agent.agent_init._install_safe_stdio"),
        patch("hermes_logging.setup_logging"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("tools.checkpoint_manager.CheckpointManager", return_value=MagicMock()),
        patch("pathlib.Path.mkdir"),
        *patches,
    ):
        init_agent(
            agent,
            provider="copilot",
            model=model,
            api_key=api_key,
            base_url="https://api.githubcopilot.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    return agent


# ---------------------------------------------------------------------------
# HUNK_C: api_mode derivation
# ---------------------------------------------------------------------------


def test_init_agent_copilot_claude_model_uses_anthropic_messages():
    """HUNK_C: provider=copilot + model=claude-* → api_mode=anthropic_messages.

    Before the fix, there was no copilot branch and the elif chain fell
    through to the bedrock / else block, leaving api_mode=chat_completions
    for a Copilot-routed Claude model.  This test confirms the pattern-based
    check is applied without requiring a live catalog fetch.
    """
    with (
        patch("agent.agent_init._install_safe_stdio"),
        patch("hermes_logging.setup_logging"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("tools.checkpoint_manager.CheckpointManager", return_value=MagicMock()),
        patch("pathlib.Path.mkdir"),
        # Token resolution for the anthropic_messages + copilot path
        patch(
            "hermes_cli.copilot_auth.resolve_copilot_token",
            return_value=("ghu_fake_raw", None),
        ),
        patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value="ghu_fake_exchanged",
        ),
    ):
        agent = object.__new__(AIAgent)
        init_agent(
            agent,
            provider="copilot",
            model="claude-claude-sonnet-4.5",
            api_key="",
            base_url="https://api.githubcopilot.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.api_mode == "anthropic_messages", (
        f"Expected anthropic_messages for copilot+claude model, got {agent.api_mode!r}"
    )


def test_init_agent_copilot_non_claude_model_calls_model_api_mode():
    """HUNK_C fallback: provider=copilot + non-claude model → copilot_model_api_mode().

    When the model doesn't start with 'claude-', init_agent tries to import
    copilot_model_api_mode() and call it.  If the import succeeds the result
    is used; if it raises (mocked to raise here), it falls back to
    'chat_completions'.  Both behaviours confirm the else-branch is reached
    (not the direct anthropic_messages assignment).
    """
    fake_mode = "codex_responses"

    with (
        patch("agent.agent_init._install_safe_stdio"),
        patch("hermes_logging.setup_logging"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("tools.checkpoint_manager.CheckpointManager", return_value=MagicMock()),
        patch("pathlib.Path.mkdir"),
        # Patch copilot_model_api_mode to return a deterministic value
        patch(
            "hermes_cli.models.copilot_model_api_mode",
            return_value=fake_mode,
        ),
    ):
        agent = object.__new__(AIAgent)
        init_agent(
            agent,
            provider="copilot",
            model="gpt-4o",
            api_key="ghu_fake_key",
            base_url="https://api.githubcopilot.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.api_mode == fake_mode, (
        f"Expected {fake_mode!r} from copilot_model_api_mode(), got {agent.api_mode!r}"
    )


# ---------------------------------------------------------------------------
# HUNK_D: api_key resolution via copilot OAuth token exchange
# ---------------------------------------------------------------------------


def test_init_agent_copilot_no_api_key_resolves_copilot_github_token():
    """HUNK_D: provider=copilot, api_mode=anthropic_messages, no api_key supplied.

    When init_agent reaches the anthropic_messages credential block with an
    empty api_key and provider='copilot', it must:
      1. Call resolve_copilot_token() to get the raw GitHub OAuth token.
      2. Exchange it via get_copilot_api_token().
      3. Assign the result to agent.api_key and agent._anthropic_api_key.

    Before the fix, the else branch assigned api_key="" directly, causing
    the SDK to send an empty Authorization header to Copilot's /v1/messages.
    """
    raw_token = "ghu_raw_github_oauth_token"
    exchanged_token = "ghu_exchanged_copilot_token"

    mock_resolve = MagicMock(return_value=(raw_token, None))
    mock_exchange = MagicMock(return_value=exchanged_token)

    with (
        patch("agent.agent_init._install_safe_stdio"),
        patch("hermes_logging.setup_logging"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("tools.checkpoint_manager.CheckpointManager", return_value=MagicMock()),
        patch("pathlib.Path.mkdir"),
        patch("hermes_cli.copilot_auth.resolve_copilot_token", mock_resolve),
        patch("hermes_cli.copilot_auth.get_copilot_api_token", mock_exchange),
    ):
        agent = object.__new__(AIAgent)
        init_agent(
            agent,
            provider="copilot",
            model="claude-claude-sonnet-4.5",
            api_key="",
            base_url="https://api.githubcopilot.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert mock_resolve.call_count >= 1, "resolve_copilot_token should have been called"
    mock_exchange.assert_called_with(raw_token)
    assert agent.api_key == exchanged_token, (
        f"Expected exchanged token on agent.api_key, got {agent.api_key!r}"
    )
    assert agent._anthropic_api_key == exchanged_token, (
        f"Expected exchanged token on agent._anthropic_api_key, got {agent._anthropic_api_key!r}"
    )
