"""The real CLI, run end to end with the scripted provider and fake sandboxes.

Covers the wiring the unit tests cannot: option parsing, the run loop, outcome
reporting, and the artifacts that land in the output directory.
"""

from __future__ import annotations

import json

from click.testing import CliRunner
from conftest import MockProvider, resp, tool
from test_swarm_verification import COMPUTED, ReproSandbox

import ctf_solver.solver.swarm as swarm_mod
from ctf_solver.cli import main


def test_cli_run_reproduces_a_flag_and_writes_artifacts(tmp_path, monkeypatch):
    challenge = tmp_path / "chall"
    challenge.mkdir()
    (challenge / "description.txt").write_text("compute the flag", encoding="utf-8")
    out = tmp_path / "out"

    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py",
                              content="# COMPUTED_FLAG_MARKER\nprint(compute())\n")]),
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
    ]})
    monkeypatch.setattr(swarm_mod, "get_provider", lambda name: provider)
    monkeypatch.setattr(swarm_mod, "HostSandbox", ReproSandbox)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--challenge-dir", str(challenge), "--claude", "1", "--no-docker", "--no-tui",
         "--max-bumps", "0", "--output-dir", str(out), "--log-dir", str(tmp_path / "logs")],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "VERIFIED FLAG" in result.output
    assert COMPUTED in result.output
    assert (out / "solve.py").is_file()
    assert "COMPUTED_FLAG_MARKER" in (out / "solve.py").read_text(encoding="utf-8")
    verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    assert verification["outcome"] == "reproduced"
    assert (out / "writeup.md").read_text(encoding="utf-8").startswith(f"# {challenge.name} 풀이 보고서")
    assert (out / "execution_log.jsonl").is_file()
    assert (out / "evidence.json").is_file()


def test_cli_unsolved_run_reports_no_verified_flag(tmp_path, monkeypatch):
    challenge = tmp_path / "chall"
    challenge.mkdir()
    (challenge / "description.txt").write_text("hard one", encoding="utf-8")
    out = tmp_path / "out"

    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("submit_flag", flag="FLAG{guess}")]),
        resp(text="I could not find anything else"),
    ]})
    monkeypatch.setattr(swarm_mod, "get_provider", lambda name: provider)
    monkeypatch.setattr(swarm_mod, "HostSandbox", ReproSandbox)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--challenge-dir", str(challenge), "--claude", "1", "--no-docker", "--no-tui",
         "--max-bumps", "0", "--output-dir", str(out), "--log-dir", str(tmp_path / "logs")],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "VERIFIED FLAG" not in result.output
    assert "No verified flag" in result.output
    assert "FLAG{guess}" in result.output  # shown as an unverified candidate
    assert not (out / "solve.py").exists()
    data = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    assert data["verified_flag"] is None
