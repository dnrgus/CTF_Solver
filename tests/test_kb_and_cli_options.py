"""KB retrieval wrapper and the CLI options added with verification and scheduling."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ctf_solver.cli import main
from ctf_solver.kb import KB_LEVEL, search_kb

REAL_KB = Path("/home/dnrgus/ctf-knowledge-base")


def _fake_kb(tmp_path: Path) -> Path:
    """A KB stub whose retriever echoes its argv, so the test can see what was passed."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "retrieve.py").write_text(
        "import sys\nprint('ARGV=' + repr(sys.argv[1:]))\n", encoding="utf-8"
    )
    return tmp_path


async def test_kb_not_configured_says_so():
    out = await search_kb("", "some evidence")
    assert "not configured" in out


async def test_kb_missing_retriever_is_reported(tmp_path):
    out = await search_kb(str(tmp_path), "evidence")
    assert "retriever not found" in out


async def test_kb_uses_level_b_and_passes_evidence_as_one_argument(tmp_path):
    kb = _fake_kb(tmp_path)
    out = await search_kb(str(kb), "amd64 ELF, NX enabled; stack overflow", category="Pwn", top=3)
    assert KB_LEVEL == "B"
    assert "unverified" in out
    assert "'--level', 'B'" in out
    assert "'--evidence', 'amd64 ELF, NX enabled; stack overflow'" in out
    assert "'--category', 'Pwn'" in out


async def test_kb_empty_evidence_is_rejected(tmp_path):
    out = await search_kb(str(_fake_kb(tmp_path)), "   ")
    assert "non-empty evidence" in out


@pytest.mark.skipif(not (REAL_KB / "tools" / "retrieve.py").is_file(), reason="KB checkout not present")
async def test_real_kb_retriever_returns_hints_at_level_b():
    out = await search_kb(str(REAL_KB), "amd64 ELF, NX enabled, PIE disabled, stack overflow", category="Pwn", top=3)
    assert out.startswith("KB hints (unverified")
    assert "failed" not in out


# --- CLI ---------------------------------------------------------------------


def test_max_agents_is_limited_to_three(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--challenge-dir", str(tmp_path), "--claude", "1",
                                  "--max-agents", "4", "--dry-run"])
    assert result.exit_code != 0
    assert "4" in result.output


def test_dry_run_accepts_verifier_and_max_agents(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--challenge-dir", str(tmp_path), "--claude", "1",
                                  "--max-agents", "2", "--verifier-cmd", "true", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output


def test_invalid_flag_pattern_is_rejected(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--challenge-dir", str(tmp_path), "--claude", "1",
                                  "--flag-pattern", "([unclosed", "--dry-run"])
    assert result.exit_code != 0
    assert "invalid flag pattern" in result.output


def test_challenge_yml_flag_pattern_is_used_when_no_cli_override(tmp_path):
    from ctf_solver.cli import _challenge_flag_pattern

    (tmp_path / "challenge.yml").write_text("name: x\nflag_pattern: 'POCTF\\{[^}]+\\}'\n", encoding="utf-8")
    assert _challenge_flag_pattern(str(tmp_path)) == r"POCTF\{[^}]+\}"
    assert _challenge_flag_pattern(str(tmp_path / "nope")) is None


def test_strict_flags_is_a_noop_with_a_notice(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--challenge-dir", str(tmp_path), "--claude", "1",
                                  "--strict-flags", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "now the default" in result.output


def test_resume_treats_unverified_legacy_flag_as_unverified(tmp_path):
    import json

    from ctf_solver.session_state import SessionState

    state = SessionState(challenge_name="x", challenge_dir=str(tmp_path), description="d", category="misc",
                         confirmed_flag="FLAG{legacy}", winner_id="claude-0")
    state.save(str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["--resume", str(tmp_path), "--dry-run"])
    assert "treated as unverified" in result.output
    assert "VERIFIED FLAG ALREADY FOUND" not in result.output
    assert json.loads((tmp_path / "session-state.json").read_text())["confirmed_flag"] == "FLAG{legacy}"
