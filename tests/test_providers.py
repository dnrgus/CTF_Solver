"""Tests for providers."""

import pytest

from ctf_solver.providers import get_provider
from ctf_solver.providers.base import SolverSession
from ctf_solver.providers.claude import ClaudeProvider, ClaudeSession
from ctf_solver.providers.codex import CodexProvider, CodexSession
from ctf_solver.providers.zai import ZAIProvider, ZAISession


def test_get_provider_claude():
    p = get_provider("claude")
    assert isinstance(p, ClaudeProvider)
    assert p.name == "claude"


def test_get_provider_codex():
    p = get_provider("codex")
    assert isinstance(p, CodexProvider)


def test_get_provider_zai():
    p = get_provider("zai")
    assert isinstance(p, ZAIProvider)


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_claude_validate_no_key():
    p = ClaudeProvider()
    assert p.validate_config({"anthropic_api_key": ""}) is False


def test_codex_validate_no_key_and_no_cli_login(monkeypatch):
    """No API key and no codex CLI login anywhere -> not configured."""
    monkeypatch.setattr("ctf_solver.providers.codex.shutil.which", lambda _: None)
    p = CodexProvider()
    assert p.validate_config({"openai_api_key": ""}) is False


def test_codex_validate_no_key_but_cli_logged_in(monkeypatch, tmp_path):
    """No API key, but `codex` is installed and has a stored login (e.g. a
    ChatGPT/Codex subscription) -> configured. _start_process() already
    inherits that same login via os.environ when no key is injected."""
    auth_dir = tmp_path / ".codex"
    auth_dir.mkdir()
    (auth_dir / "auth.json").write_text("{}")
    monkeypatch.setattr("ctf_solver.providers.codex.shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr("ctf_solver.providers.codex.Path.home", lambda: tmp_path)
    p = CodexProvider()
    assert p.validate_config({"openai_api_key": ""}) is True


def test_codex_validate_cli_present_but_never_logged_in(monkeypatch, tmp_path):
    """`codex` binary exists but no auth.json yet -> not configured."""
    monkeypatch.setattr("ctf_solver.providers.codex.shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr("ctf_solver.providers.codex.Path.home", lambda: tmp_path)
    p = CodexProvider()
    assert p.validate_config({"openai_api_key": ""}) is False


def test_zai_validate_no_key():
    p = ZAIProvider()
    assert p.validate_config({"zai_api_key": ""}) is False


def test_codex_validate_with_key():
    p = CodexProvider()
    assert p.validate_config({"openai_api_key": "sk-test"}) is True


@pytest.mark.asyncio
async def test_claude_create_session():
    p = ClaudeProvider()
    session = await p.create_session("test", "system", [], {"anthropic_api_key": "sk-test"})
    assert isinstance(session, ClaudeSession)
    assert isinstance(session, SolverSession)
    await session.close()


@pytest.mark.asyncio
async def test_codex_create_session():
    p = CodexProvider()
    session = await p.create_session("test", "system", [], {"openai_api_key": "sk-test"})
    assert isinstance(session, CodexSession)
    assert isinstance(session, SolverSession)
    await session.close()


@pytest.mark.asyncio
async def test_zai_create_session():
    p = ZAIProvider()
    session = await p.create_session("test", "system", [], {"zai_api_key": "sk-test"})
    assert isinstance(session, ZAISession)
    assert isinstance(session, SolverSession)
    await session.close()


@pytest.mark.asyncio
async def test_claude_validate_with_key():
    p = ClaudeProvider()
    assert p.validate_config({"anthropic_api_key": "sk-test"}) is True


@pytest.mark.asyncio
async def test_zai_validate_with_key():
    p = ZAIProvider()
    assert p.validate_config({"zai_api_key": "sk-test"}) is True
