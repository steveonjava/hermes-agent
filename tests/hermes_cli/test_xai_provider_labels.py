"""Regression tests for xAI provider label disambiguation."""

from types import SimpleNamespace

from hermes_cli.models import provider_label
from hermes_cli.providers import get_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label(monkeypatch):
    """The model picker must distinguish API-key and OAuth identities."""
    import agent.models_dev as models_dev

    provider = SimpleNamespace(name="Catalog API-key provider", env=["XAI_API_KEY"],
                               api="https://api.x.ai/v1", doc="")
    monkeypatch.setattr(models_dev, "get_provider_info",
                        lambda name: provider if name == "xai" else None)
    assert get_label("xai") == provider.name
    assert provider_label("xai") != provider_label("xai-oauth")
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"
    assert get_label("grok-oauth") == get_label("xai-oauth")
    assert get_label("xai-oauth") != get_label("xai")


def test_xai_label_without_catalog_uses_id_fallback(monkeypatch):
    import agent.models_dev as models_dev

    monkeypatch.setattr(models_dev, "get_provider_info", lambda name: None)
    assert get_label("xai") == "xai"
    assert get_label("xai-oauth") == get_label("grok-oauth")
    assert get_label("xai-oauth") != get_label("xai")
