from __future__ import annotations

from agent_core.observability.pricing import estimate_cost, normalize_model


def test_normalize_model_handles_snapshot_suffixes():
    assert normalize_model("openai", "gpt-4o-mini-2024-07-18") == "gpt-4o-mini"
    assert normalize_model("openai", "gpt-4o-2024-11-20") == "gpt-4o"
    assert normalize_model("anthropic", "claude-sonnet-4-20250514") == "claude-sonnet-4"


def test_estimate_cost_returns_none_for_unknown_model():
    assert estimate_cost(
        provider="openai",
        model="unknown-model",
        input_tokens=100,
        output_tokens=100,
    ) is None


def test_estimate_cost_uses_normalized_model_pricing():
    cost = estimate_cost(
        provider="openai",
        model="gpt-4o-mini-2024-07-18",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 0.75
