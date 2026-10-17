"""Non-secret runtime-binding metadata shared by resolver and lifecycle paths."""
import importlib.util

import pytest


def test_binding_roundtrip_contains_only_nonsecret_contract_fields():
    assert importlib.util.find_spec("hermes_cli.runtime_binding") is not None, (
        "Runtime binding metadata contract has not been implemented"
    )
    from hermes_cli.runtime_binding import RuntimeBinding, parse_runtime_binding, runtime_binding_state

    binding = RuntimeBinding("custom", "custom:beta", "native")
    state = runtime_binding_state(binding)
    assert state == {
        "version": 1,
        "provider": "custom",
        "requested_provider": "custom:beta",
        "credential_policy": "native",
    }
    assert parse_runtime_binding(state) == binding
    assert parse_runtime_binding(binding) is binding
    assert parse_runtime_binding(None) is None
    assert runtime_binding_state(None) is None


@pytest.mark.parametrize("changes", [
    {"version": True}, {"version": 2}, {"version": None},
    {"provider": ""}, {"provider": None}, {"requested_provider": ""},
    {"requested_provider": 23}, {"credential_policy": "unexpected"},
    {"credential_policy": True}, {"api_key": "synthetic-never-render-this"},
])
def test_binding_rejects_invalid_present_metadata_without_rendering(changes):
    from hermes_cli.runtime_binding import parse_runtime_binding

    state = {"version": 1, "provider": "custom", "requested_provider": "custom:beta", "credential_policy": "native"}
    state.update(changes)
    with pytest.raises(ValueError, match="Invalid runtime binding metadata") as caught:
        parse_runtime_binding(state)
    assert "synthetic-never-render-this" not in str(caught.value)


@pytest.mark.parametrize("value", [False, True, "native", [], {"provider": "custom"}])
def test_binding_present_malformed_is_not_legacy(value):
    from hermes_cli.runtime_binding import parse_runtime_binding

    with pytest.raises(ValueError, match="Invalid runtime binding metadata"):
        parse_runtime_binding(value)


@pytest.mark.parametrize("args", [("custom", "custom:beta", "bad"), ("", "custom:beta", "native"),
                                  ("custom", None, "native")])
def test_binding_constructor_cannot_create_invalid_authority(args):
    from hermes_cli.runtime_binding import RuntimeBinding

    with pytest.raises(ValueError, match="Invalid runtime binding metadata"):
        RuntimeBinding(*args)


@pytest.mark.parametrize(("policy", "pool_allowed", "refresh_allowed"), [
    (None, True, True), ("native", True, True), ("named_nonpool", False, True),
    ("route_override", False, True), ("key_override", False, False),
])
def test_policy_gates_do_not_infer_override_from_legacy_absence(policy, pool_allowed, refresh_allowed):
    import hermes_cli.runtime_binding as module

    assert hasattr(module, "allows_native_pool"), "Native pool permission gate is missing"
    assert hasattr(module, "allows_credential_refresh"), "Credential-source refresh gate is missing"
    value = module.RuntimeBinding("custom", "custom:beta", policy) if policy else None
    assert module.allows_native_pool(value) is pool_allowed
    assert module.allows_credential_refresh(value) is refresh_allowed
    state = module.runtime_binding_state(value)
    assert module.allows_native_pool(state) is pool_allowed
    assert module.allows_credential_refresh(state) is refresh_allowed
