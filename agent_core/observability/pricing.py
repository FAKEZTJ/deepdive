from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """USD pricing per 1M tokens."""

    input_per_million: float
    output_per_million: float


PRICING_TABLE: dict[tuple[str, str], ModelPricing] = {
    ("openai", "gpt-4o"): ModelPricing(2.50, 10.00),
    ("openai", "gpt-4o-mini"): ModelPricing(0.15, 0.60),
    ("anthropic", "claude-opus-4"): ModelPricing(15.00, 75.00),
    ("anthropic", "claude-sonnet-4"): ModelPricing(3.00, 15.00),
    ("anthropic", "claude-haiku-3.5"): ModelPricing(0.80, 4.00),
}


def normalize_model(provider: str, model: str) -> str:
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip().lower()

    if normalized_provider == "openai":
        if normalized_model.startswith("gpt-4o-mini"):
            return "gpt-4o-mini"
        if normalized_model.startswith("gpt-4o"):
            return "gpt-4o"
        return normalized_model

    if normalized_provider == "anthropic":
        if normalized_model.startswith("claude-opus-4"):
            return "claude-opus-4"
        if normalized_model.startswith("claude-sonnet-4"):
            return "claude-sonnet-4"
        if normalized_model.startswith("claude-haiku-3.5"):
            return "claude-haiku-3.5"
        return normalized_model

    return normalized_model


def estimate_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    pricing = PRICING_TABLE.get((provider.strip().lower(), normalize_model(provider, model)))
    if pricing is None:
        return None
    return (
        input_tokens * pricing.input_per_million / 1_000_000
        + output_tokens * pricing.output_per_million / 1_000_000
    )
