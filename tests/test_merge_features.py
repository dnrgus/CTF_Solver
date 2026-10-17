"""Tests for the features merged in from verialabs/ctf-agent and the A/B hooks."""

from __future__ import annotations

from unittest.mock import AsyncMock

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


def test_placeholder_detection_catches_known_sandbox_decoys():
    """Decoy flags found planted in challenge/sandbox files to catch naive
    strings/grep solving — observed live during a POCTF_2026 benchmark run
    where the solver declared victory in <60s without ever touching the
    remote target."""
    assert _is_placeholder_flag("POCTF{dev.flag.local.testing.only}")
    assert _is_placeholder_flag("POCTF{local_test_flag_padding_oracle_cbc}")


async def test_flag_rejected_without_remote_contact_when_target_has_connection_info(tmp_path, patch_backends, monkeypatch):
    """A challenge with a reachable remote target must not confirm a flag
    found after only a handful of tool calls with no evidence the solver
    ever connected to the remote host/port -- the classic decoy-grab
    pattern observed live (flag found at step 0)."""
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="CTF{found_it_locally}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, connection_info="nc 10.0.0.5 1337", claude_count=1)
    monkeypatch.setattr(swarm, "_remote_is_reachable", AsyncMock(return_value=True))

    await swarm.run()

    assert swarm.confirmed_flag is None


async def test_flag_accepted_without_remote_contact_when_remote_unreachable(tmp_path, patch_backends, monkeypatch):
    """A flag candidate must not be blocked forever just because the remote
    target happens to be down -- the contact requirement only makes sense
    when there is actually something reachable to contact."""
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="CTF{derived_locally_remote_is_down}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, connection_info="nc 10.0.0.5 1337", claude_count=1)
    monkeypatch.setattr(swarm, "_remote_is_reachable", AsyncMock(return_value=False))

    await swarm.run()

    assert swarm.confirmed_flag == "CTF{derived_locally_remote_is_down}"


async def test_flag_accepted_without_remote_contact_after_substantial_local_analysis(tmp_path, patch_backends, monkeypatch):
    """A flag found after real analysis (several tool calls), not an
    instant decoy grab, is allowed through even without remote contact."""
    provider = MockProvider({"claude-0": [
        _bash_turn("file chall.bin"),
        _bash_turn("strings chall.bin | head"),
        _bash_turn("python3 decode.py"),
        resp(tool_calls=[tool("submit_flag", flag="CTF{derived_after_real_analysis}")]),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, connection_info="nc 10.0.0.5 1337", claude_count=1)
    monkeypatch.setattr(swarm, "_remote_is_reachable", AsyncMock(return_value=True))

    await swarm.run()

    assert swarm.confirmed_flag == "CTF{derived_after_real_analysis}"


async def test_flag_accepted_after_remote_contact(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [
        _bash_turn("nc 10.0.0.5 1337"),
        resp(tool_calls=[tool("submit_flag", flag="CTF{actually_exploited_it}")]),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, connection_info="nc 10.0.0.5 1337", claude_count=1)

    await swarm.run()

    assert swarm.confirmed_flag == "CTF{actually_exploited_it}"


async def test_flag_accepted_without_remote_contact_when_no_connection_info(tmp_path, patch_backends):
    """Local-only challenges (no remote target) are unaffected by the gate."""
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="CTF{local_only_challenge}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)

    await swarm.run()

    assert swarm.confirmed_flag == "CTF{local_only_challenge}"


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
