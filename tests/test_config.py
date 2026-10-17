"""Tests for config module."""

import pytest


def test_default_settings():
    from ctf_solver.config import Settings

    s = Settings()
    assert s.sandbox_image == "ctf-sandbox"
    assert s.sandbox_memory == "4g"
    assert s.sandbox_cpus == 2
    assert s.timeout == 86400  # effectively unbounded — max_cost/max_steps are the real limits
    assert s.max_steps == 100
    assert s.max_cost == 10.0
    assert s.flag_pattern == r"([Ff][Ll][Aa][Gg]|[Cc][Tt][Ff])\{[^}]+\}"


def test_settings_from_env(monkeypatch):
    from ctf_solver.config import Settings

    monkeypatch.setenv("SANDBOX_IMAGE", "my-sandbox")
    monkeypatch.setenv("TIMEOUT", "300")
    monkeypatch.setenv("MAX_COST", "5.0")
    s = Settings()
    assert s.sandbox_image == "my-sandbox"
    assert s.timeout == 300
    assert s.max_cost == 5.0


def test_provider_counts_default():
    from ctf_solver.config import Settings

    s = Settings()
    assert s.claude_count == 0
    assert s.codex_count == 0
    assert s.zai_count == 0


def test_no_providers_raises():
    from ctf_solver.config import Settings, validate_provider_config

    s = Settings()
    with pytest.raises(ValueError, match="at least one provider"):
        validate_provider_config(s)


def test_coordinator_default_is_first_provider():
    from ctf_solver.config import Settings, get_coordinator_provider

    s = Settings(claude_count=0, codex_count=2, zai_count=1)
    assert get_coordinator_provider(s) == "codex"


def test_coordinator_zero_count_excluded():
    from ctf_solver.config import Settings, get_coordinator_provider

    s = Settings(claude_count=0, codex_count=3)
    assert get_coordinator_provider(s) == "codex"


def test_coordinator_explicit_overrides():
    from ctf_solver.config import Settings, get_coordinator_provider

    s = Settings(claude_count=2, codex_count=2, coordinator="codex")
    assert get_coordinator_provider(s) == "codex"

# --------------------------------------------------------------------------- #
# codex_model default: smart fallback, but any explicit setting still wins
# --------------------------------------------------------------------------- #
def test_codex_cli_config_model_reads_model_line(tmp_path, monkeypatch):
    from ctf_solver.config import _codex_cli_config_model

    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model = "gpt-5.6-sol"\nmodel_reasoning_effort = "medium"\n')

    assert _codex_cli_config_model() == "gpt-5.6-sol"


def test_codex_cli_config_model_ignores_other_model_prefixed_keys(tmp_path, monkeypatch):
    """model_reasoning_effort (etc.) must not be mistaken for the model= line."""
    from ctf_solver.config import _codex_cli_config_model

    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model_reasoning_effort = "medium"\nmodel = "gpt-6-astra"\n')

    assert _codex_cli_config_model() == "gpt-6-astra"


def test_codex_cli_config_model_none_when_missing(tmp_path, monkeypatch):
    from ctf_solver.config import _codex_cli_config_model

    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    assert _codex_cli_config_model() is None


def test_default_codex_model_uses_cli_config_without_api_key(tmp_path, monkeypatch):
    from ctf_solver.config import _default_codex_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model = "gpt-5.6-sol"\n')

    assert _default_codex_model() == "gpt-5.6-sol"


def test_default_codex_model_ignores_cli_config_with_api_key(tmp_path, monkeypatch):
    """An OPENAI_API_KEY means billed API calls, which accept the hardcoded
    default regardless — the CLI's own subscription model shouldn't apply."""
    from ctf_solver.config import _default_codex_model

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model = "gpt-5.6-sol"\n')

    assert _default_codex_model() == "gpt-5.4"


def test_default_codex_model_falls_back_with_nothing_available(tmp_path, monkeypatch):
    from ctf_solver.config import _default_codex_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)

    assert _default_codex_model() == "gpt-5.4"


def test_explicit_codex_model_env_var_still_wins(tmp_path, monkeypatch):
    """The actual point: nothing about the smarter default removes the
    ability to configure it explicitly — CODEX_MODEL always overrides."""
    from ctf_solver.config import Settings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_MODEL", "my-explicit-choice")
    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model = "gpt-5.6-sol"\n')

    assert Settings().codex_model == "my-explicit-choice"


def test_settings_codex_model_picks_up_cli_config_when_unset(tmp_path, monkeypatch):
    from ctf_solver.config import Settings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    monkeypatch.setattr("ctf_solver.config.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model = "gpt-5.6-sol"\n')

    assert Settings().codex_model == "gpt-5.6-sol"
