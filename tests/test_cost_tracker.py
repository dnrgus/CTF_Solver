"""Tests for cost tracker."""

from ctf_solver.tracking.cost_tracker import (
    _GENAI_PRICES_AVAILABLE,
    CostTracker,
    _calc_cost_fallback,
)


def test_record_and_total():
    ct = CostTracker()
    ct.record_tokens("solver-1", "claude-opus-4-6", input_tokens=1000, output_tokens=500, provider_spec="claude-sdk")
    expected = (1000 * 5.0 + 500 * 25.0) / 1_000_000
    assert abs(ct.total_cost_usd - expected) < 0.001
    assert ct.total_tokens == 1500


def test_multiple_agents():
    ct = CostTracker()
    ct.record_tokens("s1", "gpt-5.4", input_tokens=2000, output_tokens=1000, provider_spec="codex")
    ct.record_tokens("s2", "gpt-5.4-mini", input_tokens=500, output_tokens=200, provider_spec="codex")
    assert ct.total_tokens == 3700


def test_format_usage():
    ct = CostTracker()
    ct.record_tokens("s1", "claude-opus-4-6", input_tokens=50000, output_tokens=5000, provider_spec="claude-sdk")
    formatted = ct.format_usage("s1")
    assert "$" in formatted


def test_max_cost_exceeded():
    ct = CostTracker()
    ct.record_tokens("s1", "gpt-5.4", input_tokens=100000, output_tokens=50000, provider_spec="codex")
    assert ct.is_over_budget(0.01)


# --- genai-prices integration (the preferred pricing path) -----------------


def test_genai_prices_used_for_known_model():
    """gpt-4o lives in the genai-prices DB, so the genai path (not the local
    fallback table, which has no gpt-4o entry) must produce the cost."""
    assert _GENAI_PRICES_AVAILABLE, "genai-prices is a declared dependency"
    ct = CostTracker()
    ct.record_tokens("s1", "gpt-4o", input_tokens=1000, output_tokens=500, provider_spec="openai")
    cost = ct.total_cost_usd
    assert cost > 0
    # Hand calc with public gpt-4o rates: $2.50/Mtok in, $10.00/Mtok out.
    expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
    assert abs(cost - expected) < expected * 0.5  # tolerant: rates can drift


def test_glm_5_1_nonzero_via_genai_prices():
    """The local FALLBACK table prices glm-5.1 at 0, but genai-prices has real
    GLM pricing. A non-zero cost proves genai-prices takes precedence."""
    assert _GENAI_PRICES_AVAILABLE
    # Sanity: the fallback table really would yield 0 for this model.
    assert _calc_cost_fallback(1000, 500, 0, "glm-5.1") == 0.0
    ct = CostTracker()
    ct.record_tokens("s1", "glm-5.1", input_tokens=1000, output_tokens=500, provider_spec="zai")
    assert ct.total_cost_usd > 0


def test_unknown_model_falls_back_no_crash():
    """No genai match and no fallback-table match → 0.0, and no exception."""
    ct = CostTracker()
    ct.record_tokens(
        "s1", "totally-made-up-model-zzz", input_tokens=1000, output_tokens=500, provider_spec="x"
    )
    assert ct.total_cost_usd == 0.0
    assert ct.total_tokens == 1500


def test_cache_discount_cheaper_than_uncached():
    """For a known model, the same input_tokens cost less when part is a cache read."""
    full = CostTracker()
    full.record_tokens("s1", "gpt-4o", input_tokens=10000, output_tokens=0, provider_spec="openai")

    cached = CostTracker()
    cached.record_tokens(
        "s1", "gpt-4o", input_tokens=10000, output_tokens=0, cache_read_tokens=8000,
        provider_spec="openai",
    )

    assert cached.total_cost_usd < full.total_cost_usd
    assert cached.total_cost_usd > 0


def test_calc_cost_fallback_direct():
    """Exercise the local table math directly, including the prefix-match path."""
    # Exact key match: claude-opus-4-6 → $5/Mtok in, $25/Mtok out.
    direct = _calc_cost_fallback(1000, 500, 0, "claude-opus-4-6")
    assert direct == (1000 * 5.00 + 500 * 25.00) / 1_000_000

    # Cache-read subset is billed at the cached rate ($0.50/Mtok), remainder full.
    cached = _calc_cost_fallback(1000, 500, 400, "claude-opus-4-6")
    expected_cached = (600 * 5.00 + 400 * 0.50 + 500 * 25.00) / 1_000_000
    assert cached == expected_cached
    assert cached < direct

    # Prefix match: a dated alias starting with a table key reuses its pricing.
    prefixed = _calc_cost_fallback(1000, 500, 0, "claude-opus-4-6-20991231")
    assert prefixed == direct

    # No match anywhere → 0.0.
    assert _calc_cost_fallback(1000, 500, 0, "no-such-model-xyz") == 0.0
