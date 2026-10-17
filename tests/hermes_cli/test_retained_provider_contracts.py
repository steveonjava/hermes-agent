"""End-to-end contracts retained from legacy custom-provider routes."""

from pathlib import Path

import pytest

from hermes_cli.config import (
    _custom_provider_entry_to_provider_config,
    atomic_config_replace,
    read_raw_config,
)
from hermes_cli.config_migrations import run_migrations
from hermes_cli.runtime_provider import resolve_runtime_provider


@pytest.fixture(autouse=True)
def offline_worktree_resolution(monkeypatch):
    from hermes_cli import config
    from hermes_cli import runtime_provider

    root = Path(__file__).resolve().parents[2]
    for module in (config, runtime_provider):
        assert Path(module.__file__).resolve().is_relative_to(root)

    def reject_network(*args, **kwargs):
        raise AssertionError("Provider contract tests must not open network connections")

    monkeypatch.setattr("socket.socket.connect", reject_network)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **k: 128_000)


def test_legacy_capabilities_survive_conversion_migration_and_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    capabilities = {"openai_native_compaction": True, "vision": False}
    legacy = {
        "name": "Retained Relay", "base_url": "https://relay.example/v1",
        "model": "gpt-5.6", "api_key": "test-only-key",
        "api_mode": "codex_responses", "capabilities": capabilities,
    }
    converted = _custom_provider_entry_to_provider_config(legacy)
    assert converted["capabilities"] == capabilities
    atomic_config_replace(tmp_path / "config.yaml", {"_config_version": 11, "custom_providers": [legacy]})
    results = {"config_added": [], "warnings": []}
    run_migrations(11, results, quiet=True)
    assert not results.get("warnings")
    persisted = read_raw_config()
    assert "custom_providers" not in persisted
    assert persisted["providers"]["retained-relay"]["capabilities"] == capabilities
    runtime = resolve_runtime_provider(requested="retained-relay")
    assert runtime["capabilities"] == capabilities
    assert runtime["model"] == "gpt-5.6"
    assert runtime["api_mode"] == "codex_responses"
