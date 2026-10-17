"""Validation of explicit transport trust without restricting generic capability names."""
from typing import Any

TRUST_WIRES = {
    "openai_native_compaction": "codex_responses",
    "anthropic_tool_cache_marker": "anthropic_messages",
    "anthropic_signed_thinking": "anthropic_messages",
}


def validate_capabilities(value: Any, api_mode: str | None = None) -> dict[str, bool]:
    """Validate trust before filtering metadata; non-dicts retain upstream empty-map behavior.

    This is not a general-name allowlist: arbitrary string-keyed Boolean flags survive.
    """
    if not isinstance(value, dict):
        return {}
    for flag in TRUST_WIRES:
        if flag in value and not isinstance(value[flag], bool):
            raise ValueError(f"Capability {flag} must be a Boolean")
    if api_mode is not None:
        from hermes_cli.config_providers import _canonical_api_mode
        wire = _canonical_api_mode(api_mode).lower()
        for flag, required in TRUST_WIRES.items():
            if value.get(flag) is True and wire != required:
                raise ValueError(f"Capability {flag} requires {required}")
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, bool)}


def configured_capability_wire(entry: dict, model: str | None = None) -> str | None:
    """Use the runtime's declared, model-derived, or URL-inferred custom wire."""
    from hermes_cli.config_providers import _canonical_api_mode
    from hermes_cli.runtime_provider import _detect_api_mode_for_url
    mode = entry.get("api_mode") or entry.get("transport")
    if isinstance(mode, str) and mode.strip():
        return _canonical_api_mode(mode).lower()
    url = entry.get("base_url") or entry.get("api") or entry.get("url") or ""
    from hermes_cli.runtime_provider_custom import _opencode_family_for_custom
    family = _opencode_family_for_custom(entry.get("name") or "", url)
    effective_model = model or entry.get("model") or entry.get("default_model")
    if family:
        if not effective_model:
            # Runtime owns target/global model selection; only wire checks must wait.
            return None
        from hermes_cli.models import opencode_model_api_mode
        return opencode_model_api_mode(family, effective_model)
    return _detect_api_mode_for_url(url) or "chat_completions"
