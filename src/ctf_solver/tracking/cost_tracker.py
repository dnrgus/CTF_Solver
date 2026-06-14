"""Per-agent token/cost tracking with genai-prices and fallback pricing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from genai_prices import Usage as _GPUsage
    from genai_prices import calc_price as _gp_calc_price

    _GENAI_PRICES_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is declared, guard is defensive
    _GENAI_PRICES_AVAILABLE = False

FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 5.00, "cached_input": 0.50, "output": 25.00},
    "claude-opus-4-6-20250612": {"input": 5.00, "cached_input": 0.50, "output": 25.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.3-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "glm-5.1": {"input": 0.0, "cached_input": 0.0, "output": 0.0},
}


def _calc_cost(input_tokens: int, output_tokens: int, cache_read_tokens: int, model: str) -> float:
    """Price a call. Prefer the genai-prices DB; fall back to the local table.

    genai-prices treats ``input_tokens`` as the total prompt size with
    ``cache_read_tokens`` a discounted subset of it — the same convention this
    tracker uses — so the token fields pass through unchanged.
    """
    if _GENAI_PRICES_AVAILABLE and model:
        try:
            price = _gp_calc_price(
                _GPUsage(
                    input_tokens=input_tokens,
                    cache_read_tokens=cache_read_tokens,
                    output_tokens=output_tokens,
                ),
                model_ref=model,
            )
            return float(price.total_price)
        except Exception:  # unknown model / lookup failure → local table
            pass
    return _calc_cost_fallback(input_tokens, output_tokens, cache_read_tokens, model)


def _calc_cost_fallback(input_tokens: int, output_tokens: int, cache_read_tokens: int, model: str) -> float:
    if not model:
        return 0.0
    pricing = FALLBACK_PRICING.get(model)
    if not pricing:
        for key, val in FALLBACK_PRICING.items():
            if model.startswith(key) or key.startswith(model):
                pricing = val
                break
    if not pricing:
        return 0.0
    input_rate = pricing["input"]
    cached_rate = pricing.get("cached_input", input_rate)
    output_rate = pricing["output"]
    uncached = max(0, input_tokens - cache_read_tokens)
    return (uncached * input_rate + cache_read_tokens * cached_rate + output_tokens * output_rate) / 1_000_000


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


@dataclass
class AgentUsage:
    model_name: str = ""
    provider_spec: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    by_agent: dict[str, AgentUsage] = field(default_factory=dict)

    def record_tokens(
        self,
        agent_name: str,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        provider_spec: str = "",
    ) -> None:
        cost = _calc_cost(input_tokens, output_tokens, cache_read_tokens, model_name)
        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = AgentUsage(model_name=model_name, provider_spec=provider_spec)
        agent = self.by_agent[agent_name]
        agent.input_tokens += input_tokens
        agent.output_tokens += output_tokens
        agent.cache_read_tokens += cache_read_tokens
        agent.cost_usd += cost

    @property
    def total_cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.by_agent.values())

    @property
    def total_tokens(self) -> int:
        return sum(a.input_tokens + a.output_tokens for a in self.by_agent.values())

    def is_over_budget(self, max_cost: float) -> bool:
        return self.total_cost_usd >= max_cost

    def format_usage(self, agent_name: str) -> str:
        agent = self.by_agent.get(agent_name)
        if not agent:
            return ""
        return (
            f"{_fmt_tokens(agent.input_tokens)} in / "
            f"{_fmt_tokens(agent.output_tokens)} out | "
            f"${agent.cost_usd:.4f}"
        )
