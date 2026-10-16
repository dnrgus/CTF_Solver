"""Tests for the features merged in from verialabs/ctf-agent and the A/B hooks."""

from __future__ import annotations

from conftest import MockProvider, resp, tool
from test_swarm_behavioral import _bash_turn, _swarm

import ctf_solver.solver.swarm as swarm_mod
from ctf_solver.prompts import ChallengeMeta, build_prompt
from ctf_solver.providers.base import TokenUsage
from ctf_solver.solver.solver_base import ResultStatus
from ctf_solver.solver.swarm import _is_placeholder_flag


def _fast_bump(monkeypatch):
    monkeypatch.setattr(swarm_mod, "BUMP_COOLDOWN_STEP", 0.01)
    monkeypatch.setattr(swarm_mod, "BUMP_COOLDOWN_MAX", 0.01)


async def test_bump_resumes_stopped_solver_and_solves(tmp_path, patch_backends, monkeypatch):
    _fast_bump(monkeypatch)
    provider = MockProvider({"claude-0": [
        _bash_turn("ls"),
        resp(text="I give up"),
        resp(tool_calls=[tool("submit_flag", flag="FLAG{second_wind}")]),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=2)

    result = await swarm.run()

    assert result is not None and result.status == ResultStatus.SOLVED
    assert result.flag == "FLAG{second_wind}"
    sent = provider.sessions["claude-0"].sent
    assert any("different approach" in msg for msg, _ in sent)


async def test_bump_disabled_with_zero_max_bumps(tmp_path, patch_backends, monkeypatch):
    _fast_bump(monkeypatch)
    provider = MockProvider({"claude-0": [
        _bash_turn("ls"),
        resp(text="I give up"),
        resp(tool_calls=[tool("submit_flag", flag="FLAG{never_reached}")]),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=0)

    result = await swarm.run()

    assert result is None
    assert swarm.confirmed_flag is None


async def test_broken_solver_is_not_bumped(tmp_path, patch_backends, monkeypatch):
    _fast_bump(monkeypatch)
    provider = MockProvider({"claude-0": [resp(text="nothing", usage=TokenUsage())]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=2)

    await swarm.run()

    assert len(provider.sessions["claude-0"].sent) == 1


async def test_placeholder_flag_is_rejected(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{flag}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)

    await swarm.run()

    assert swarm.confirmed_flag is None


def test_placeholder_detection():
    assert _is_placeholder_flag("CTF{flag}")
    assert _is_placeholder_flag("FLAG{placeholder}")
    assert not _is_placeholder_flag("FLAG{r34l_fl4g_9f2}")


async def test_strict_flags_never_confirms(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{real_one}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, accept_regex_flag_as_verified=False)

    await swarm.run()

    assert swarm.confirmed_flag is None


def test_prompt_nc_heredoc_and_first_action():
    meta = ChallengeMeta(name="n", category="pwn", connection_info="nc localhost 1337")
    prompt = build_prompt(meta, [])
    assert "FIRST ACTION REQUIRED" in prompt
    assert "<<'EOF'" in prompt
    assert "host.docker.internal" in prompt


def test_prompt_web_connection_hint():
    meta = ChallengeMeta(name="w", category="web", connection_info="http://x.test/")
    assert "web service" in build_prompt(meta, [])


def test_prompt_image_mentions_view_image():
    meta = ChallengeMeta(name="s", category="stego")
    assert "view_image" in build_prompt(meta, ["a.png"])


def test_prompt_pyghidra_only_for_binary_categories():
    assert "pyghidra" in build_prompt(ChallengeMeta(name="r", category="reverse"), [])
    assert "pyghidra" not in build_prompt(ChallengeMeta(name="w", category="web"), [])


def test_prompt_harness_appended_last():
    prompt = build_prompt(ChallengeMeta(name="h"), [], harness="RULE: always verify")
    assert prompt.rstrip().endswith("RULE: always verify")
    assert "## Harness" in prompt
    assert "## Harness" not in build_prompt(ChallengeMeta(name="h"), [])
