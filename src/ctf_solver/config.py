"""Pydantic Settings — credentials and configuration from .env + env vars + CLI."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# yakisoba's own hardcoded default. Still used whenever an OPENAI_API_KEY is
# configured — fine for a billed API key, which accepts arbitrary model ids.
_API_KEY_DEFAULT_CODEX_MODEL = "gpt-5.4"


def _codex_cli_config_model() -> str | None:
    """Whatever `~/.codex/config.toml`'s own `model =` line already says.

    That's the model the user's `codex` CLI uses every day, so a ChatGPT/Codex
    subscription login (ANTHROPIC_API_KEY/OPENAI_API_KEY not set) is already
    known to accept it — unlike the hardcoded "gpt-5.4" default, which a
    ChatGPT-plan login rejects outright.
    """
    cfg = Path.home() / ".codex" / "config.toml"
    if not cfg.exists():
        return None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("model") and "=" in line and not line.startswith("model_"):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _default_codex_model() -> str:
    """Only a *default* — CODEX_MODEL env var, `--config` TOML, or any other
    explicit setting still always wins; this factory never runs when one of
    those supplies the field."""
    if os.environ.get("OPENAI_API_KEY"):
        return _API_KEY_DEFAULT_CODEX_MODEL
    return _codex_cli_config_model() or _API_KEY_DEFAULT_CODEX_MODEL


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    zai_api_key: str = ""
    zai_endpoint: str = "https://api.z.ai/api/paas/v4/"
    claude_model: str = "claude-opus-4-6-20250612"
    codex_model: str = Field(default_factory=_default_codex_model)
    zai_model: str = "glm-5.1"

    # Provider counts (set via CLI, not env)
    claude_count: int = 0
    codex_count: int = 0
    zai_count: int = 0
    coordinator: str = ""
    no_coordinator: bool = False

    # Sandbox
    sandbox_image: str = "ctf-sandbox"
    sandbox_memory: str = "4g"
    sandbox_cpus: int = 2
    no_docker: bool = False

    # Limits
    # No per-challenge wall-clock cap by default — matches verialabs' own
    # "solvers never give up" design this project is based on; yakisoba's
    # own spec had baked in a hard 600s default that cuts a solver off mid
    # -exploit regardless of real progress (confirmed live: a genuine,
    # on-track OT/Modbus exploit got killed by this at the 10-minute mark).
    # --max-cost/--max-steps/--max-bumps are the real backstops against a
    # runaway solver; this is only a last-resort ceiling, not a working
    # limit anyone should expect to hit.
    timeout: int = 86400  # 24h
    max_steps: int = 100
    max_cost: float = 10.0
    flag_pattern: str = r"([Ff][Ll][Aa][Gg]|[Cc][Tt][Ff])\{[^}]+\}"

    # Hints
    # No external judge in manual mode, so a pattern match is the only check.
    # --strict-flags turns this off (candidates are recorded but never confirmed).
    accept_regex_flag_as_verified: bool = True
    max_bumps: int = 2
    harness_file: str = ""
    hint: str = ""
    interactive: bool = False

    # Output
    output_dir: str = ""
    log_dir: str = ""
    port: int = 0

    # Modes
    no_tui: bool = False
    dry_run: bool = False
    verbose: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def to_provider_config(self) -> dict:
        return {
            "anthropic_api_key": self.anthropic_api_key,
            "openai_api_key": self.openai_api_key,
            "zai_api_key": self.zai_api_key,
            "zai_endpoint": self.zai_endpoint,
            "claude_model": self.claude_model,
            "codex_model": self.codex_model,
            "zai_model": self.zai_model,
        }


def validate_provider_config(settings: Settings) -> None:
    """Validate that at least one provider has a positive count."""
    if settings.claude_count + settings.codex_count + settings.zai_count <= 0:
        msg = "Specify at least one provider (--claude N, --codex N, or --zai N)"
        raise ValueError(msg)


def get_active_providers(settings: Settings) -> list[tuple[str, int]]:
    """Return list of (provider_name, count) for providers with positive count."""
    result = []
    if settings.claude_count > 0:
        result.append(("claude", settings.claude_count))
    if settings.codex_count > 0:
        result.append(("codex", settings.codex_count))
    if settings.zai_count > 0:
        result.append(("zai", settings.zai_count))
    return result


def get_coordinator_provider(settings: Settings) -> str | None:
    """Get the coordinator provider name, or None if --no-coordinator."""
    if settings.no_coordinator:
        return None
    if settings.coordinator:
        active = {name for name, _ in get_active_providers(settings)}
        if settings.coordinator not in active:
            msg = f"Coordinator '{settings.coordinator}' has no active solvers. Active: {active}"
            raise ValueError(msg)
        return settings.coordinator
    providers = get_active_providers(settings)
    return providers[0][0] if providers else None


def load_toml_config(path: str) -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)
