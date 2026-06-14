"""Writeup generator — produces solve-detail.md and solve-brief.md.

The detailed writeup is reconstructed from the winning solver's JSONL trace
file (``SolverResult.trace_path``). Each line of that file is a JSON object
emitted by :class:`ctf_solver.tracing.SolverTracer`; the relevant event types
are:

- ``tool_call``   — ``{"type", "tool", "args", "step", ...}``
- ``tool_result`` — ``{"type", "tool", "result", "step", ...}`` (truncated)
- ``usage``       — ``{"type", "input_tokens", "output_tokens", ...}``

Parsing is best-effort: a missing, empty, or partially written trace simply
causes the trace-derived sections to be omitted rather than raising.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ctf_solver.solver.solver_base import ResultStatus, SolverResult

# Per-tool argument key used to summarise a call in the human-readable narrative.
_TOOL_ARG_KEYS: dict[str, str] = {
    "bash": "command",
    "read_file": "path",
    "write_file": "path",
    "list_files": "path",
    "submit_flag": "flag",
    "view_image": "filename",
    "web_fetch": "url",
    "webhook_get_requests": "uuid",
    "notify_coordinator": "message",
}

# Max number of tool calls to render verbatim in the Raw Trace section.
_RAW_TRACE_LIMIT = 40
# Max characters of a single argument value shown in the Approach narrative.
_ARG_PREVIEW = 120
# Max characters of a tool result shown in the Raw Trace section.
_RESULT_PREVIEW = 400


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _arg_summary(tool: str, args: Any) -> str:
    """Return a short, human-readable summary of a tool call's arguments."""
    if isinstance(args, dict):
        key = _TOOL_ARG_KEYS.get(tool)
        if key and key in args and args[key] not in (None, ""):
            return _truncate(str(args[key]), _ARG_PREVIEW)
        # Fall back to whichever string-ish value is present.
        for value in args.values():
            if isinstance(value, (str, int, float)) and str(value):
                return _truncate(str(value), _ARG_PREVIEW)
        return ""
    if args in (None, ""):
        return ""
    return _truncate(str(args), _ARG_PREVIEW)


def _parse_trace(trace_path: Path | str | None) -> list[dict[str, Any]]:
    """Parse a JSONL trace file into a list of event dicts.

    Returns an empty list if the file is missing/unreadable. Lines that fail to
    parse as JSON (e.g. a partial final line) are skipped silently.
    """
    if not trace_path:
        return []
    path = Path(trace_path)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _build_tool_counts(events: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ev in events:
        if ev.get("type") == "tool_call":
            tool = ev.get("tool")
            if tool:
                counts[str(tool)] += 1
    return counts


def _build_approach(events: list[dict[str, Any]]) -> str:
    """Reconstruct an ordered, numbered narrative from tool_call events."""
    lines: list[str] = []
    n = 0
    for ev in events:
        if ev.get("type") != "tool_call":
            continue
        n += 1
        tool = str(ev.get("tool", "?"))
        summary = _arg_summary(tool, ev.get("args"))
        if summary:
            lines.append(f"{n}. **{tool}** — `{summary}`")
        else:
            lines.append(f"{n}. **{tool}**")
    if not lines:
        return ""
    return "\n".join(lines)


def _find_flag_extraction(events: list[dict[str, Any]], flag: str | None) -> str:
    """Describe how/where the flag first surfaced in the trace."""
    if not flag:
        return ""
    # Prefer an explicit submit_flag call carrying the winning flag.
    for ev in events:
        if ev.get("type") == "tool_call" and str(ev.get("tool")) == "submit_flag":
            args = ev.get("args")
            submitted = args.get("flag") if isinstance(args, dict) else None
            if submitted and flag in str(submitted):
                step = ev.get("step")
                where = f" at step {step}" if step is not None else ""
                return f"Submitted via the `submit_flag` tool{where}: `{flag}`."
    # Otherwise look for the flag inside any tool result output.
    for ev in events:
        if ev.get("type") == "tool_result" and flag in str(ev.get("result", "")):
            tool = str(ev.get("tool", "a tool"))
            step = ev.get("step")
            where = f" at step {step}" if step is not None else ""
            snippet = _truncate(str(ev.get("result", "")), _RESULT_PREVIEW)
            return (
                f"Appeared in the output of `{tool}`{where}:\n\n"
                f"```\n{snippet}\n```"
            )
    # Otherwise it was seen in an LLM/event payload (or the trace is partial).
    for ev in events:
        if ev.get("type") not in ("tool_call", "tool_result"):
            for value in ev.values():
                if isinstance(value, str) and flag in value:
                    return f"Identified in the solver's reasoning output: `{flag}`."
    return f"Flag `{flag}` was captured by the solver (exact origin not recorded in the trace)."


def _build_raw_trace(events: list[dict[str, Any]]) -> str:
    """Render key tool calls paired with their (truncated) outputs."""
    # Index tool results by step for pairing with their originating call.
    results_by_step: dict[Any, dict[str, Any]] = {}
    for ev in events:
        if ev.get("type") == "tool_result":
            results_by_step[ev.get("step")] = ev

    blocks: list[str] = []
    shown = 0
    for ev in events:
        if ev.get("type") != "tool_call":
            continue
        if shown >= _RAW_TRACE_LIMIT:
            blocks.append("_… trace truncated …_")
            break
        shown += 1
        tool = str(ev.get("tool", "?"))
        step = ev.get("step")
        summary = _arg_summary(tool, ev.get("args"))
        header = f"**[{step}] {tool}**" if step is not None else f"**{tool}**"
        block = header
        if summary:
            block += f"\n```\n{summary}\n```"
        result_ev = results_by_step.get(step)
        if result_ev is not None:
            out = _truncate(str(result_ev.get("result", "")), _RESULT_PREVIEW)
            if out:
                block += f"\n→\n```\n{out}\n```"
        blocks.append(block)
    return "\n\n".join(blocks)


def generate_writeup(
    output_dir: str,
    challenge_name: str,
    category: str,
    result: SolverResult,
    total_cost_usd: float = 0.0,
    total_solvers: int = 1,
) -> None:
    """Write ``solve-detail.md`` and ``solve-brief.md`` into ``output_dir``.

    The detailed report's narrative sections are reconstructed from
    ``result.trace_path`` when available; if the trace is missing or partial,
    those sections degrade gracefully (header/metadata are always written).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    flag_display = result.flag or "No flag found"
    status_text = "SOLVED" if result.status == ResultStatus.SOLVED else result.status.value

    events = _parse_trace(result.trace_path)

    sections: list[str] = [
        f"# Challenge: {challenge_name}",
        (
            f"- **Flag**: `{flag_display}`\n"
            f"- **Category**: {category}\n"
            f"- **Status**: {status_text}\n"
            f"- **Winner**: {result.solver_id} "
            f"({result.steps} steps, {_format_duration(result.duration)}, ${result.cost_usd:.2f})\n"
            f"- **Total Cost**: ${total_cost_usd:.2f} across {total_solvers} solvers"
        ),
    ]

    approach = _build_approach(events)
    if approach:
        sections.append("## Approach\n" + approach)
    elif result.findings_summary:
        # No usable trace, but we still have a findings summary to show.
        sections.append("## Approach\n" + result.findings_summary)

    tool_counts = _build_tool_counts(events)
    if tool_counts:
        rendered = ", ".join(
            f"{tool} ({count} call{'s' if count != 1 else ''})"
            for tool, count in tool_counts.most_common()
        )
        sections.append("## Tools Used\n" + rendered)

    flag_extraction = _find_flag_extraction(events, result.flag)
    if flag_extraction:
        sections.append("## Flag Extraction\n" + flag_extraction)

    if result.findings_summary and approach:
        # Surface findings as well when we already showed a trace-based approach.
        sections.append("## Findings\n" + result.findings_summary)

    raw_trace = _build_raw_trace(events)
    if raw_trace:
        sections.append("## Raw Trace\n" + raw_trace)

    # Always point at the underlying trace file for reference.
    sections.append(f"## Trace File\n`{result.trace_path}`")

    detail = "\n\n".join(sections) + "\n"
    (out / "solve-detail.md").write_text(detail)

    _write_brief(out, challenge_name, result)


def _write_brief(out: Path, challenge_name: str, result: SolverResult) -> None:
    """Write the short 3-5 sentence ``solve-brief.md`` summary."""
    summary = (result.findings_summary or "").strip()
    if result.flag:
        body = summary or "See solve-detail.md for the full writeup."
        brief = (
            f"# {challenge_name} — Flag: `{result.flag}`\n\n"
            f"{_truncate(body, 600)}\n"
        )
    else:
        body = summary or "No solution was found."
        brief = (
            f"# {challenge_name} — Unsolved\n\n"
            f"{_truncate(body, 600)}\n"
        )
    (out / "solve-brief.md").write_text(brief)
