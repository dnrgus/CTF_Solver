"""Run artifacts are derived from recorded state and never overstate the outcome."""

from __future__ import annotations

import json

from conftest import MockProvider, resp, tool
from test_swarm_behavioral import _bash_turn, _swarm
from test_swarm_verification import COMPUTED, use_repro_sandbox

from ctf_solver.artifacts import build_verification, build_writeup, write_run_artifacts


def _read_verification(tmp_path) -> dict:
    return json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))


async def test_reproduced_run_publishes_the_exact_reproduced_script(tmp_path, patch_backends, monkeypatch):
    script = "# COMPUTED_FLAG_MARKER\nprint(compute())\n"
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py", content=script)]),
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
    ]})
    patch_backends(provider)
    use_repro_sandbox(monkeypatch)
    swarm = _swarm(tmp_path, claude_count=1)
    result = await swarm.run()
    assert result is not None

    names = write_run_artifacts(swarm, "test", "misc")

    assert "solve.py" in names
    assert (tmp_path / "solve.py").read_text(encoding="utf-8") == script
    data = _read_verification(tmp_path)
    assert data["outcome"] == "reproduced"
    assert data["verified_flag"] == COMPUTED
    assert data["verified_status"] == "locally_reproduced"
    assert data["verified_by"] == "claude-0"
    assert data["verifier"]["configured"] is False
    assert any(f["flag"] == COMPUTED and f["status"] == "locally_reproduced" for f in data["flags"])


async def test_writeup_is_korean_and_reports_the_reproduction(tmp_path, patch_backends, monkeypatch):
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py",
                              content="# COMPUTED_FLAG_MARKER\n")]),
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
    ]})
    patch_backends(provider)
    use_repro_sandbox(monkeypatch)
    swarm = _swarm(tmp_path, claude_count=1)
    await swarm.run()

    text = build_writeup(swarm, "test", "misc")

    assert "풀이 보고서" in text
    assert "풀이 완료 (로컬 재현 확인" in text
    assert "재현한 스크립트" in text
    assert COMPUTED in text
    assert "한계 및 미검증 사항" in text


async def test_unsolved_run_has_only_a_labelled_draft(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py", content="print('wip')\n")]),
        resp(text="gave up"),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=0)
    await swarm.run()

    names = write_run_artifacts(swarm, "test", "misc")

    assert "solve.py" not in names
    assert "solve_candidate.py" in names
    text = (tmp_path / "writeup.md").read_text(encoding="utf-8")
    assert "미해결" in text
    assert "재현되지 않았습니다" in text


async def test_verifier_only_solve_is_not_described_as_reproduced(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py", content="print('x')\n")]),
        resp(tool_calls=[tool("submit_flag", flag="FLAG{by_verifier}")]),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, verifier_cmd="true")
    await swarm.run()

    text = build_writeup(swarm, "test", "misc")
    data = build_verification(swarm, "test", "misc")

    assert data["outcome"] == "solved"
    assert data["verified_status"] == "remote_confirmed"
    assert "로컬 재현은 수행되지 않았고" in text
    assert "풀이 완료 (검증기 확인)" in text


async def test_unverified_candidates_never_appear_as_verified(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("submit_flag", flag="FLAG{maybe}")]),
        _bash_turn("more"),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=0)
    await swarm.run()

    data = build_verification(swarm, "test", "misc")

    assert data["verified_flag"] is None
    assert data["outcome"] == "unsolved"
    flags = {f["flag"]: f for f in data["flags"]}
    assert flags["FLAG{maybe}"]["status"] == "candidate"
    assert any("검증된 플래그가 없으므로" in item for item in data["limitations"])
