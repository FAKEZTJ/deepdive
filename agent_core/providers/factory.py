from __future__ import annotations

import os

from agent_core.providers.anthropic_provider import AnthropicProvider
from agent_core.providers.base import LLMProvider, ProviderConfig
from agent_core.providers.deepseek_provider import DeepSeekProvider
from agent_core.providers.openai_provider import OpenAIProvider

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4",
    "deepseek": "deepseek-chat",
}

API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def build_provider(
    provider_name: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    normalized_name = provider_name.strip().lower()
    if normalized_name not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported provider '{provider_name}'.")
    resolved_api_key = api_key or os.getenv(API_KEY_ENV_VARS.get(normalized_name, ""))
    if not resolved_api_key:
        env_var = API_KEY_ENV_VARS.get(normalized_name, "API_KEY")
        raise ValueError(f"Missing API key for provider '{normalized_name}'. Set {env_var}.")

    config = ProviderConfig(
        model=model or DEFAULT_MODELS[normalized_name],
        api_key=resolved_api_key,
        base_url=base_url,
    )
    if normalized_name == "openai":
        return OpenAIProvider(config)
    if normalized_name == "anthropic":
        return AnthropicProvider(config)
    if normalized_name == "deepseek":
        return DeepSeekProvider(config)
    raise AssertionError(f"Unhandled provider '{provider_name}'.")
