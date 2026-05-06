from __future__ import annotations

from agent_core.providers.deepseek_provider import DeepSeekProvider
from agent_core.providers.factory import build_provider
from agent_core.providers.base import ProviderConfig


def test_deepseek_provider_sets_openai_compatible_default_base_url():
    provider = DeepSeekProvider(ProviderConfig(model="deepseek-chat", api_key="test-key"))

    assert provider.name == "deepseek"
    assert provider.config.base_url == "https://api.deepseek.com/v1"


def test_build_provider_can_construct_deepseek_provider():
    provider = build_provider("deepseek", api_key="test-key")

    assert isinstance(provider, DeepSeekProvider)
    assert provider.config.model == "deepseek-chat"
