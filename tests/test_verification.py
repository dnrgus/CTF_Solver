"""Verification states, the operator-supplied verifier, and flag format rules."""

from __future__ import annotations

import pytest

from ctf_solver.tools.flag import DEFAULT_FLAG_PATTERN, extract_flags, is_exact_flag
from ctf_solver.verification import (
    TERMINAL_OK,
    CommandVerifier,
    FlagRecord,
    FlagStatus,
)

# --- flag format -------------------------------------------------------------


def test_default_pattern_does_not_match_inside_a_word():
    """'xflag{...}' and 'myCTF{...}' are not flags; the old pattern found them."""
    assert extract_flags("xflag{abc} and myCTF{def}") == []


def test_default_pattern_extracts_standalone_flag_from_prose():
    assert extract_flags("Found: FLAG{hello}.") == ["FLAG{hello}"]


def test_exact_match_requires_the_whole_string():
    assert is_exact_flag("FLAG{abc}", DEFAULT_FLAG_PATTERN)
    assert not is_exact_flag("junk FLAG{abc} junk", DEFAULT_FLAG_PATTERN)
    assert not is_exact_flag("FLAG{abc}\nmore", DEFAULT_FLAG_PATTERN)


def test_per_competition_format_is_exact_and_not_the_default():
    """POCTF{...} is a different format: the default does not accept it, the explicit pattern does."""
    pattern = r"POCTF\{[^}]+\}"
    assert not is_exact_flag("POCTF{abc}", DEFAULT_FLAG_PATTERN)
    assert is_exact_flag("POCTF{abc}", pattern)
    assert not is_exact_flag("FLAG{abc}", pattern)


# --- candidate records -------------------------------------------------------


def test_candidate_is_not_verified():
    record = FlagRecord(flag="FLAG{x}", solver_id="a", step=1)
    assert record.status == FlagStatus.CANDIDATE
    assert not record.verified


@pytest.mark.parametrize("status", sorted(TERMINAL_OK, key=lambda s: s.value))
def test_only_reproduced_or_remote_confirmed_are_terminal(status):
    record = FlagRecord(flag="FLAG{x}", solver_id="a", step=1, status=status)
    assert record.verified


@pytest.mark.parametrize("status", [FlagStatus.CANDIDATE, FlagStatus.UNKNOWN, FlagStatus.FAILED])
def test_non_terminal_statuses_are_not_verified(status):
    assert not FlagRecord(flag="FLAG{x}", solver_id="a", step=1, status=status).verified


def test_verifier_rejection_is_final_but_reproduction_failure_is_not():
    verifier_no = FlagRecord(flag="F", solver_id="a", step=1, status=FlagStatus.FAILED, source="verifier")
    repro_no = FlagRecord(flag="F", solver_id="a", step=1, status=FlagStatus.FAILED, source="reproduction")
    assert verifier_no.finally_rejected
    assert not repro_no.finally_rejected


# --- operator verifier -------------------------------------------------------


async def test_verifier_exit_zero_is_remote_confirmed():
    status, note = await CommandVerifier("true").check("FLAG{ok}")
    assert status == FlagStatus.REMOTE_CONFIRMED
    assert "accepted" in note


async def test_verifier_exit_one_is_rejected():
    status, _ = await CommandVerifier("false").check("FLAG{no}")
    assert status == FlagStatus.FAILED


async def test_verifier_other_exit_codes_are_unknown_not_rejected():
    """A scoreboard outage must not be read as a wrong flag."""
    status, note = await CommandVerifier("exit 3").check("FLAG{x}")
    assert status == FlagStatus.UNKNOWN
    assert "error" in note


async def test_verifier_receives_flag_in_environment_not_in_command(tmp_path):
    marker = tmp_path / "seen.txt"
    command = f'printf "%s" "$CTF_FLAG" > {marker}'
    status, _ = await CommandVerifier(command).check("FLAG{env_channel}")
    assert status == FlagStatus.REMOTE_CONFIRMED
    assert marker.read_text() == "FLAG{env_channel}"
    assert "FLAG{env_channel}" not in command


async def test_verifier_timeout_is_unknown(monkeypatch):
    verifier = CommandVerifier("sleep 5", timeout_s=0.2)
    status, note = await verifier.check("FLAG{slow}")
    assert status == FlagStatus.UNKNOWN
    assert "timed out" in note


def test_verifier_requires_a_command():
    with pytest.raises(ValueError):
        CommandVerifier("   ")
