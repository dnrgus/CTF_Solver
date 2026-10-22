"""Run artifacts: verification.json, solve.py, writeup.md, evidence.json.

Everything here is derived from the swarm's recorded state (flag records, the
evidence ledger, per-agent results). Nothing is written from a model's
self-report, so the writeup cannot describe a path the run did not take.

- ``solve.py`` is written only when a flag was verified. It is the exact script
  that reproduced the flag (or, for a verifier-confirmed flag, the agent's own
  solve.py at the end of its run, with the verification noted). An unverified
  run gets ``solve_candidate.py``, labelled as a draft.
- ``verification.json`` is the machine-readable status of every candidate.
- ``evidence.json`` summarises the ledger; ``execution_log.jsonl`` is the ledger
  itself and is appended to during the run.
- ``writeup.md`` is in Korean and is a report of the recorded state.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from ctf_solver.evidence import KIND_HYPOTHESIS, EvidenceLedger
from ctf_solver.verification import FlagStatus

if TYPE_CHECKING:
    from ctf_solver.solver.swarm import ChallengeSwarm

VERIFICATION_SCHEMA = 1

_OUTCOME_BY_STATUS = {
    FlagStatus.REMOTE_CONFIRMED: "solved",
    FlagStatus.LOCALLY_REPRODUCED: "reproduced",
}
_STATUS_LABEL_KO = {
    "candidate": "후보 (미검증)",
    "locally_reproduced": "로컬 재현 성공",
    "remote_confirmed": "검증기 확인",
    "failed": "실패",
    "unknown": "검증 불가",
}
_OUTCOME_LABEL_KO = {
    "solved": "풀이 완료 (검증기 확인)",
    "reproduced": "풀이 완료 (로컬 재현 확인, 공식 검증기 미설정)",
    "unsolved": "미해결 (검증된 플래그 없음)",
}


def _outcome(swarm: ChallengeSwarm) -> str:
    if swarm.confirmed_status is None:
        return "unsolved"
    return _OUTCOME_BY_STATUS.get(swarm.confirmed_status, "unsolved")


def _limitations(swarm: ChallengeSwarm, outcome: str) -> list[str]:
    items = [
        "LLM 발언(findings)은 검증되지 않은 주장이며 사실로 취급하지 않는다.",
        "로컬 재현은 solve.py를 새 샌드박스에서 다시 실행한 결과이며, 원격 서비스가 필요한 풀이는 재실행 시점에 대상이 살아 있어야 한다.",
    ]
    if swarm.settings.verifier_cmd.strip():
        items.append("검증기(--verifier-cmd)의 판정은 해당 명령이 권위 있는 채점기와 일치한다는 전제에서만 유효하다.")
    else:
        items.append("공식 검증기가 설정되지 않아 REMOTE_CONFIRMED 상태는 기록되지 않는다.")
    if outcome == "unsolved":
        items.append("검증된 플래그가 없으므로 이 실행의 성공을 주장하지 않는다.")
    return items


def _agent_rows(swarm: ChallengeSwarm) -> list[dict]:
    rows = []
    for sid, inst in swarm.solvers.items():
        result = swarm._results.get(sid)
        rows.append({
            "solver_id": sid,
            "provider": inst.provider_name,
            "status": result.value if result else ("running" if not inst.finished else "done"),
            "steps": inst.step_count,
            "cost_usd": round(inst.cost_usd, 6),
            "verified_flag": inst.flag,
        })
    return rows


def build_verification(swarm: ChallengeSwarm, challenge_name: str, category: str) -> dict:
    outcome = _outcome(swarm)
    verified_owner = next((i for i in swarm.solvers.values() if i.confirmed), None)
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "challenge": challenge_name,
        "category": category,
        "generated_at": round(time.time(), 3),
        "outcome": outcome,
        "verified_flag": swarm.confirmed_flag,
        "verified_status": swarm.confirmed_status.value if swarm.confirmed_status else None,
        "verified_by": verified_owner.solver_id if verified_owner else None,
        "verifier": {
            "configured": bool(swarm.settings.verifier_cmd.strip()),
            "note": "command contents are not recorded (they may contain credentials)",
        },
        "flags": [record.to_dict() for record in swarm.flag_records.values()],
        "agents": _agent_rows(swarm),
        "limitations": _limitations(swarm, outcome),
    }


def _solve_script(swarm: ChallengeSwarm) -> tuple[str | None, str]:
    """Return (source, filename). Only a reproduced/verified script is named solve.py."""
    owner = next((i for i in swarm.solvers.values() if i.confirmed), None)
    if owner is not None and owner.verified_source:
        return owner.verified_source, "solve.py"
    if owner is not None and owner.solve_source:
        return owner.solve_source, "solve.py"
    drafts = [i for i in swarm.solvers.values() if i.solve_source]
    if drafts:
        return drafts[0].solve_source, "solve_candidate.py"
    return None, ""


def _fmt_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (없음)"


def _hypothesis_lines(ledger: EvidenceLedger) -> list[str]:
    lines: list[str] = []
    for rec in ledger.records:
        if rec.get("kind") != KIND_HYPOTHESIS:
            continue
        verdict = rec.get("verdict", "")
        line = f"{len(lines) + 1}. [{rec.get('agent', '?')}, 단계 {rec.get('step', '?')}] {rec.get('hypothesis', '')} → **{verdict}**"
        if rec.get("evidence"):
            line += f"\n   - 근거: {rec['evidence']}"
        if rec.get("failure_cause"):
            line += f"\n   - 실패 원인: {rec['failure_cause']}"
        if rec.get("next_step"):
            line += f"\n   - 다음 실험: {rec['next_step']}"
        lines.append(line)
    return lines


def build_writeup(swarm: ChallengeSwarm, challenge_name: str, category: str) -> str:
    outcome = _outcome(swarm)
    ledger = swarm.evidence
    summary = ledger.summary()
    flag_rows = [
        f"| `{r.flag}` | {_STATUS_LABEL_KO.get(r.status.value, r.status.value)} | {r.solver_id} | "
        f"{(r.notes[-1] if r.notes else '').replace('|', '/')[:160]} |"
        for r in swarm.flag_records.values()
    ]
    agent_rows = [
        f"| {row['solver_id']} | {row['provider']} | {row['steps']} | ${row['cost_usd']:.4f} | {row['status']} |"
        for row in _agent_rows(swarm)
    ]
    _, filename = _solve_script(swarm)
    owner = next((i for i in swarm.solvers.values() if i.confirmed), None)
    if filename == "solve.py" and owner is not None and owner.verified_source:
        solve_section = (
            "`solve.py`는 새 샌드박스에서 실행해 플래그를 재현한 스크립트입니다(재현 근거: "
            "`verification.json`의 `flags[].notes`)."
        )
    elif filename == "solve.py":
        solve_section = (
            "`solve.py`는 에이전트가 남긴 스크립트입니다. 로컬 재현은 수행되지 않았고, 플래그는 "
            "공식 검증기가 확인했습니다. 이 스크립트가 독립적으로 같은 플래그를 출력한다는 근거는 없습니다."
        )
    elif filename == "solve_candidate.py":
        solve_section = (
            "검증된 플래그가 없어 `solve.py`를 만들지 않았습니다. 에이전트가 남긴 스크립트는 "
            "`solve_candidate.py`에 **초안**으로만 보관했으며 재현되지 않았습니다."
        )
    else:
        solve_section = "에이전트가 `solve.py`를 작성하지 않았습니다. 풀이 스크립트가 없어 재현할 수 없습니다."

    parts = [
        f"# {challenge_name} 풀이 보고서",
        "",
        "## 결과",
        f"- 상태: **{_OUTCOME_LABEL_KO[outcome]}**",
        f"- 검증된 플래그: `{swarm.confirmed_flag}`" if swarm.confirmed_flag else "- 검증된 플래그: 없음",
        f"- 분류: {category or '미지정'}",
        f"- 분석 기록: 도구 호출 {summary['bash_calls']}회 (bash), 억제 {summary['suppressed_calls']}회, "
        f"실패 명령 반복 {summary['repeat_failures']}회",
        "",
        "## 플래그 후보와 검증 상태",
        "| 플래그 | 상태 | 제출 에이전트 | 최신 근거 |",
        "|---|---|---|---|",
        *(flag_rows or ["| (후보 없음) | | | |"]),
        "",
        "## 에이전트",
        "| 에이전트 | 제공자 | 도구 단계 | 비용 | 결과 |",
        "|---|---|---|---|---|",
        *agent_rows,
        "",
        "## 분석 과정 (기록된 가설과 실험)",
        *(_hypothesis_lines(ledger) or ["기록된 가설이 없습니다."]),
        "",
        "## 관찰된 사실 (시스템 기록: 종료 코드 0인 명령)",
        _fmt_list(summary["observations"]),
        "",
        "## 풀이 스크립트와 재현",
        solve_section,
        "",
        "## 미검증 주장 (에이전트 발언, 참고용)",
        _fmt_list(summary["claims"]),
        "",
        "## 한계 및 미검증 사항",
        _fmt_list(_limitations(swarm, outcome)),
        "",
    ]
    return "\n".join(parts)


def write_run_artifacts(swarm: ChallengeSwarm, challenge_name: str, category: str) -> list[str]:
    """Write every artifact to the swarm's artifact directory. Returns the file names written."""
    out = swarm.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def write(name: str, text: str) -> None:
        (out / name).write_text(text, encoding="utf-8")
        written.append(name)

    verification = build_verification(swarm, challenge_name, category)
    write("verification.json", json.dumps(verification, indent=2, ensure_ascii=False, default=str))
    swarm.evidence.save_summary(out / "evidence.json")
    written.append("evidence.json")
    if not (out / "execution_log.jsonl").exists():
        (out / "execution_log.jsonl").touch()
    written.append("execution_log.jsonl")
    source, filename = _solve_script(swarm)
    if source and filename:
        write(filename, source if source.endswith("\n") else source + "\n")
    write("writeup.md", build_writeup(swarm, challenge_name, category))
    return written
