"""Tests for writeup generator."""

import json
from pathlib import Path

from ctf_solver.solver.solver_base import ResultStatus, SolverResult
from ctf_solver.writeup import generate_writeup


def _write_trace(tmp_path: Path, events: list[dict]) -> Path:
    """Write a fake JSONL trace and return its path."""
    trace = tmp_path / "trace.jsonl"
    with trace.open("w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return trace


def _sample_events() -> list[dict]:
    return [
        {"type": "tool_call", "tool": "bash", "args": {"command": "ls -la /challenge"}, "step": 1},
        {"type": "tool_result", "tool": "bash", "result": "vuln  flag.txt", "step": 1},
        {"type": "tool_call", "tool": "read_file", "args": {"path": "/challenge/vuln.c"}, "step": 2},
        {"type": "tool_result", "tool": "read_file", "result": "int main() { gets(buf); }", "step": 2},
        {"type": "tool_call", "tool": "bash", "args": {"command": "./exploit.py"}, "step": 3},
        {"type": "tool_result", "tool": "bash", "result": "you win: FLAG{pwned}", "step": 3},
        {"type": "usage", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cost_usd": 0.01},
        {"type": "tool_call", "tool": "submit_flag", "args": {"flag": "FLAG{pwned}"}, "step": 4},
    ]


def test_generate_detail(tmp_path):
    result = SolverResult(
        solver_id="claude-0",
        status=ResultStatus.SOLVED,
        flag="FLAG{test}",
        steps=47,
        duration=154.0,
        findings_summary="Found buffer overflow at 0x0804...",
    )
    generate_writeup(str(tmp_path), "baby-pwn", "pwn", result, total_cost_usd=5.67, total_solvers=3)
    detail = (tmp_path / "solve-detail.md").read_text()
    assert "FLAG{test}" in detail
    assert "baby-pwn" in detail
    assert "claude-0" in detail
    assert "$5.67 across 3 solvers" in detail
    # 154s -> 2m 34s
    assert "2m 34s" in detail


def test_generate_brief(tmp_path):
    result = SolverResult(
        solver_id="codex-0",
        status=ResultStatus.SOLVED,
        flag="CTF{win}",
        steps=20,
        duration=60.0,
        findings_summary="Simple XOR encoding",
    )
    generate_writeup(str(tmp_path), "easy-crypto", "crypto", result, total_cost_usd=1.00, total_solvers=5)
    brief = (tmp_path / "solve-brief.md").read_text()
    assert "CTF{win}" in brief
    assert "easy-crypto" in brief
    assert "Simple XOR encoding" in brief


def test_no_flag_skips(tmp_path):
    result = SolverResult(
        solver_id="zai-0",
        status=ResultStatus.FAILED,
        flag=None,
        steps=10,
        duration=30.0,
        findings_summary="Could not solve",
    )
    generate_writeup(str(tmp_path), "hard-rev", "rev", result, total_cost_usd=2.00, total_solvers=3)
    detail = (tmp_path / "solve-detail.md").read_text()
    assert "No flag found" in detail
    brief = (tmp_path / "solve-brief.md").read_text()
    assert "Unsolved" in brief


def test_detail_sections_from_trace(tmp_path):
    trace = _write_trace(tmp_path, _sample_events())
    result = SolverResult(
        solver_id="claude-1",
        status=ResultStatus.SOLVED,
        flag="FLAG{pwned}",
        steps=4,
        duration=90.0,
        cost_usd=0.34,
        trace_path=trace,
        findings_summary="gets() overflow gives shell",
    )
    generate_writeup(str(tmp_path), "trace-pwn", "pwn", result, total_cost_usd=1.23, total_solvers=2)
    detail = (tmp_path / "solve-detail.md").read_text()

    # Header / metadata
    assert "# Challenge: trace-pwn" in detail
    assert "FLAG{pwned}" in detail
    assert "claude-1 (4 steps, 1m 30s, $0.34)" in detail

    # Required sections present
    assert "## Approach" in detail
    assert "## Tools Used" in detail
    assert "## Flag Extraction" in detail
    assert "## Raw Trace" in detail

    # Approach narrative reflects ordered tool calls with arg summaries
    assert "1. **bash**" in detail
    assert "ls -la /challenge" in detail
    assert "/challenge/vuln.c" in detail

    # Tools Used counts: bash called twice, read_file once, submit_flag once
    assert "bash (2 calls)" in detail
    assert "read_file (1 call)" in detail
    assert "submit_flag (1 call)" in detail

    # Flag Extraction should reference the submit_flag tool path
    flag_section = detail.split("## Flag Extraction", 1)[1]
    assert "submit_flag" in flag_section
    assert "step 4" in flag_section

    # Raw Trace shows step-indexed calls and outputs
    assert "[1] bash" in detail
    assert "you win: FLAG{pwned}" in detail


def test_flag_extraction_from_tool_output(tmp_path):
    # No submit_flag call: flag only appears in a tool_result output.
    events = [
        {"type": "tool_call", "tool": "bash", "args": {"command": "cat flag.txt"}, "step": 1},
        {"type": "tool_result", "tool": "bash", "result": "FLAG{from_output}", "step": 1},
    ]
    trace = _write_trace(tmp_path, events)
    result = SolverResult(
        solver_id="codex-1",
        status=ResultStatus.SOLVED,
        flag="FLAG{from_output}",
        steps=1,
        duration=5.0,
        trace_path=trace,
    )
    generate_writeup(str(tmp_path), "cat-chall", "misc", result)
    detail = (tmp_path / "solve-detail.md").read_text()
    flag_section = detail.split("## Flag Extraction", 1)[1]
    assert "`bash`" in flag_section
    assert "FLAG{from_output}" in flag_section


def test_missing_trace_degrades_gracefully(tmp_path):
    # trace_path points nowhere -> trace-derived sections omitted, no crash.
    result = SolverResult(
        solver_id="zai-1",
        status=ResultStatus.SOLVED,
        flag="FLAG{notrace}",
        steps=3,
        duration=12.0,
        trace_path=tmp_path / "does-not-exist.jsonl",
        findings_summary="Solved without a recorded trace.",
    )
    generate_writeup(str(tmp_path), "no-trace", "misc", result)
    detail = (tmp_path / "solve-detail.md").read_text()
    assert "FLAG{notrace}" in detail
    # Falls back to findings_summary for the Approach section.
    assert "## Approach" in detail
    assert "Solved without a recorded trace." in detail
    # No trace -> no Tools Used / Raw Trace sections.
    assert "## Tools Used" not in detail
    assert "## Raw Trace" not in detail


def test_partial_trace_is_robust(tmp_path):
    # A truncated final line (partial JSON) must be skipped, not crash.
    trace = tmp_path / "partial.jsonl"
    good = json.dumps({"type": "tool_call", "tool": "bash", "args": {"command": "id"}, "step": 1})
    trace.write_text(good + "\n" + '{"type": "tool_call", "tool": "bash"')  # truncated line
    result = SolverResult(
        solver_id="claude-2",
        status=ResultStatus.SOLVED,
        flag="FLAG{partial}",
        steps=1,
        duration=3.0,
        trace_path=trace,
    )
    generate_writeup(str(tmp_path), "partial", "misc", result)
    detail = (tmp_path / "solve-detail.md").read_text()
    assert "## Tools Used" in detail
    assert "bash (1 call)" in detail
    assert "1. **bash**" in detail
