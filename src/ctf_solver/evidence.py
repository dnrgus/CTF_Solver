"""Structured evidence ledger: hypotheses, commands, outputs, failures, next steps.

``execution_log.jsonl`` is the source of truth. It is append-only, one record per
line, and every state change goes through ``_append`` -> ``_apply``. Loading a
log replays the same ``_apply`` path, so an interrupted run recovers its refuted
hypotheses, candidate flags, failed commands and novelty history exactly.

What it is used for:

- suppression: an identical bash command that failed is not re-run unless new
  evidence has arrived since (the ``epoch`` counter advances on novel output,
  flag events and recorded tool results only, never on unverified LLM text).
- stall detection: an agent whose outputs have stopped being novel is stalled.
- handoff: a new agent, a bumped agent or a resumed run gets the refuted
  hypotheses, candidates, verified observations and file paths. Unverified
  claims from other agents are included but labelled as claims.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

KIND_TOOL = "tool"
KIND_SUPPRESSED = "suppressed"
KIND_HYPOTHESIS = "hypothesis"
KIND_FLAG = "flag"
KIND_CLAIM = "claim"
KIND_SCHEDULER = "scheduler"

OUTPUT_HEAD_CHARS = 300
COMMAND_CHARS = 2000
MAX_OBSERVATIONS = 12
MAX_CLAIMS = 6
MAX_PATHS = 40
HANDOFF_MAX_CHARS = 2500

_EXIT_RE = re.compile(r"\[exit (-?\d+)\]\s*$")
_PATH_RE = re.compile(r"/challenge/[\w./-]+")
VERDICTS = ("supported", "refuted", "inconclusive")


def command_key(command: str) -> str:
    """Identity of a command for failure lookups: whitespace-insensitive."""
    normalized = " ".join(command.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def digest_output(output: str) -> str:
    return hashlib.sha1(output.encode("utf-8", errors="replace")).hexdigest()


def exit_code_of(output: str) -> int:
    """``do_bash`` appends ``[exit N]`` when the command failed; absent means 0."""
    match = _EXIT_RE.search(output.rstrip())
    return int(match.group(1)) if match else 0


class EvidenceLedger:
    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.epoch = 0
        self.records: list[dict[str, Any]] = []
        self.refuted: list[dict[str, Any]] = []
        self.observations: list[str] = []
        self.claims: list[str] = []
        self.paths: set[str] = set()
        self.flag_status: dict[str, dict[str, str]] = {}
        self.spawns: list[dict[str, Any]] = []
        self.bash_calls = 0
        self.suppressed_calls = 0
        self.repeat_failures = 0
        self._digests: set[str] = set()
        self._failed: dict[str, dict[str, Any]] = {}

    # ---- persistence -------------------------------------------------------

    @classmethod
    def load(cls, log_path: Path | str) -> EvidenceLedger:
        """Rebuild a ledger by replaying its log. A missing log gives an empty ledger."""
        ledger = cls(log_path)
        path = Path(log_path)
        if not path.is_file():
            return ledger
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial last line from a killed process
            if isinstance(rec, dict) and "kind" in rec:
                ledger.records.append(rec)
                ledger._apply(rec)
        return ledger

    def _append(self, rec: dict[str, Any]) -> dict[str, Any]:
        full = {"seq": len(self.records) + 1, "ts": round(time.time(), 3), **rec}
        self._apply(full)
        self.records.append(full)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(full, ensure_ascii=False, default=str) + "\n")
        return full

    # ---- single mutation point (live and replay) ---------------------------

    def _apply(self, rec: dict[str, Any]) -> None:
        kind = rec.get("kind")
        if kind == KIND_TOOL:
            self._apply_tool(rec)
        elif kind == KIND_SUPPRESSED:
            self.suppressed_calls += 1
        elif kind == KIND_HYPOTHESIS:
            self._apply_hypothesis(rec)
        elif kind == KIND_FLAG:
            self.epoch += 1
            self.flag_status[rec["flag"]] = {
                "status": rec["status"], "agent": rec.get("agent", ""), "source": rec.get("source", ""),
            }
            if rec["status"] in ("remote_confirmed", "locally_reproduced"):
                self._observe(f"flag {rec['flag']} -> {rec['status']} ({rec.get('note', '')[:120]})")
        elif kind == KIND_CLAIM:
            if len(self.claims) < MAX_CLAIMS * 4:
                self.claims.append(f"[{rec.get('agent', '?')}, step {rec.get('step', '?')}] {rec['text']}")
        elif kind == KIND_SCHEDULER:
            self.spawns.append({"agent": rec.get("agent", ""), "reason": rec.get("reason", "")})

    def _apply_tool(self, rec: dict[str, Any]) -> None:
        command = rec.get("command")
        if rec.get("novel"):
            self._digests.add(rec["digest"])
            self.epoch += 1
        if command is None:
            return
        self.bash_calls += 1
        self.paths.update(_PATH_RE.findall(command))
        code = int(rec.get("exit_code", 0))
        key = command_key(command)
        if code != 0:
            if key in self._failed:
                self.repeat_failures += 1
            self._failed[key] = {
                "epoch": self.epoch,
                "exit_code": code,
                "output_head": rec.get("output_head", ""),
                "step": rec.get("step"),
                "agent": rec.get("agent", ""),
            }
        elif rec.get("novel"):
            head = " ".join(str(rec.get("output_head", "")).split())[:160]
            self._observe(f"`{' '.join(command.split())[:160]}` -> {head or '(no output)'}")

    def _apply_hypothesis(self, rec: dict[str, Any]) -> None:
        if rec.get("verdict") == "refuted":
            self.refuted.append({
                "hypothesis": rec.get("hypothesis", ""),
                "agent": rec.get("agent", ""),
                "step": rec.get("step"),
                "failure_cause": rec.get("failure_cause", ""),
                "next_step": rec.get("next_step", ""),
            })
        elif rec.get("verdict") == "supported":
            self.claims.append(f"[{rec.get('agent', '?')}, step {rec.get('step', '?')}] supported: {rec.get('hypothesis', '')}")

    def _observe(self, text: str) -> None:
        self.observations.append(text)
        del self.observations[:-MAX_OBSERVATIONS]

    # ---- recording API used by the swarm -----------------------------------

    def record_tool(
        self, *, agent: str, step: int, tool: str, args: dict | str,
        output: str, command: str | None = None,
    ) -> bool:
        """Record one executed tool call. Returns True when its output was new evidence."""
        digest = digest_output(output)
        novel = digest not in self._digests
        rec: dict[str, Any] = {
            "kind": KIND_TOOL, "agent": agent, "step": step, "tool": tool,
            "digest": digest, "novel": novel,
            "output_head": output.strip()[:OUTPUT_HEAD_CHARS],
        }
        if command is not None:
            rec["command"] = command[:COMMAND_CHARS]
            rec["exit_code"] = exit_code_of(output)
        else:
            rec["args"] = str(args)[:OUTPUT_HEAD_CHARS]
        self._append(rec)
        return novel

    def record_suppressed(self, *, agent: str, step: int, command: str) -> None:
        self._append({"kind": KIND_SUPPRESSED, "agent": agent, "step": step,
                      "command": command[:COMMAND_CHARS]})

    def record_hypothesis(
        self, *, agent: str, step: int, hypothesis: str, verdict: str,
        evidence: str = "", failure_cause: str = "", next_step: str = "",
    ) -> None:
        if verdict not in VERDICTS:
            msg = f"verdict must be one of {VERDICTS}, got {verdict!r}"
            raise ValueError(msg)
        self._append({
            "kind": KIND_HYPOTHESIS, "agent": agent, "step": step,
            "hypothesis": hypothesis[:500], "verdict": verdict,
            "evidence": evidence[:1000], "failure_cause": failure_cause[:500],
            "next_step": next_step[:500],
        })

    def record_flag(
        self, *, agent: str, step: int, flag: str, status: str, source: str = "", note: str = "",
    ) -> None:
        self._append({"kind": KIND_FLAG, "agent": agent, "step": step, "flag": flag,
                      "status": status, "source": source, "note": note[:500]})

    def record_claim(self, *, agent: str, step: int, text: str) -> None:
        """An LLM's own statement. Stored for handoff, never counted as evidence."""
        self._append({"kind": KIND_CLAIM, "agent": agent, "step": step, "text": text[:400]})

    def record_scheduler(self, *, agent: str, reason: str) -> None:
        self._append({"kind": KIND_SCHEDULER, "agent": agent, "reason": reason})

    # ---- queries ------------------------------------------------------------

    def suppression(self, command: str) -> str | None:
        """A note to return instead of re-running ``command``, or None to run it."""
        failed = self._failed.get(command_key(command))
        if failed is None or failed["epoch"] != self.epoch:
            return None
        head = failed["output_head"] or "(no output)"
        return (
            f"[suppressed] This exact command already failed (exit {failed['exit_code']}) at "
            f"step {failed['step']} and nothing new has been observed since. Previous output:\n"
            f"{head}\n"
            "Change the input or the approach before retrying."
        )

    def handoff_brief(self, max_chars: int = HANDOFF_MAX_CHARS) -> str:
        """The current state in a compact form for a new or resumed agent."""
        lines: list[str] = ["## Recorded state (machine-recorded facts vs. unverified claims)"]
        if self.flag_status:
            lines.append("Flag candidates:")
            lines += [f"- {flag}: {info['status']}" for flag, info in self.flag_status.items()]
        if self.refuted:
            lines.append("Refuted hypotheses (do not retry without new evidence):")
            for item in self.refuted[-MAX_CLAIMS * 2:]:
                cause = f" — cause: {item['failure_cause']}" if item["failure_cause"] else ""
                lines.append(f"- {item['hypothesis']}{cause}")
        if self.observations:
            lines.append("Observed (command exited 0, recorded by the system):")
            lines += [f"- {obs}" for obs in self.observations[-MAX_OBSERVATIONS:]]
        if self.claims:
            lines.append("Unverified claims from other agents (may be wrong):")
            lines += [f"- {c}" for c in self.claims[-MAX_CLAIMS:]]
        if self.paths:
            lines.append("Paths touched: " + ", ".join(sorted(self.paths)[:MAX_PATHS]))
        if len(lines) == 1:
            return ""
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 15] + "\n… [truncated]"

    def summary(self) -> dict[str, Any]:
        return {
            "record_count": len(self.records),
            "bash_calls": self.bash_calls,
            "suppressed_calls": self.suppressed_calls,
            "repeat_failures": self.repeat_failures,
            "epoch": self.epoch,
            "flag_status": self.flag_status,
            "refuted_hypotheses": self.refuted,
            "observations": self.observations,
            "claims": self.claims[-MAX_CLAIMS * 4:],
            "paths": sorted(self.paths)[:MAX_PATHS],
            "spawns": self.spawns,
        }

    def save_summary(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
