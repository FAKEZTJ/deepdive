from __future__ import annotations

from typing import Any

from agent_core.providers.base import ProviderConfig
from agent_core.providers.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"

    def __init__(self, config: ProviderConfig, client: Any | None = None):
        effective_config = config.model_copy(
            update={"base_url": config.base_url or "https://api.deepseek.com/v1"}
        )
        super().__init__(effective_config, client=client)
