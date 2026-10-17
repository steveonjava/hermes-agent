"""Non-secret route selection and credential-policy metadata.

This record carries authority, never credentials. Native identity resolution and
endpoint scoping remain the resolver/lifecycle caller's responsibility.
"""
from dataclasses import dataclass
from typing import Any


_POLICIES = frozenset({"native", "named_nonpool", "route_override", "key_override"})
_STATE_KEYS = frozenset({"version", "provider", "requested_provider", "credential_policy"})


def _validate_fields(provider: Any, requested_provider: Any, credential_policy: Any) -> None:
    if (not isinstance(provider, str) or not provider.strip()
            or not isinstance(requested_provider, str) or not requested_provider.strip()
            or not isinstance(credential_policy, str) or credential_policy not in _POLICIES):
        raise ValueError("Invalid runtime binding metadata")


@dataclass(frozen=True)
class RuntimeBinding:
    provider: str
    requested_provider: str
    credential_policy: str = "native"

    def __post_init__(self) -> None:
        _validate_fields(self.provider, self.requested_provider, self.credential_policy)


def parse_runtime_binding(value: Any) -> RuntimeBinding | None:
    """None is legacy absence; any malformed present authority fails closed."""
    if value is None:
        return None
    if isinstance(value, RuntimeBinding):
        _validate_fields(value.provider, value.requested_provider, value.credential_policy)
        return value
    if (not isinstance(value, dict) or set(value) != _STATE_KEYS
            or type(value.get("version")) is not int or value["version"] != 1):
        raise ValueError("Invalid runtime binding metadata")
    return RuntimeBinding(value["provider"], value["requested_provider"], value["credential_policy"])


def runtime_binding_state(value: Any) -> dict[str, Any] | None:
    binding = parse_runtime_binding(value)
    if binding is None:
        return None
    return {
        "version": 1,
        "provider": binding.provider,
        "requested_provider": binding.requested_provider,
        "credential_policy": binding.credential_policy,
    }


def allows_native_pool(value: Any) -> bool:
    """Legacy absence preserves native discovery; detached policies never do."""
    binding = parse_runtime_binding(value)
    return binding is None or binding.credential_policy == "native"


def allows_credential_refresh(value: Any) -> bool:
    """An actual key pin forbids substitution from a different credential source."""
    binding = parse_runtime_binding(value)
    return binding is None or binding.credential_policy != "key_override"
