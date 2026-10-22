"""Knowledge-base hints via the KB's own deterministic retriever.

The KB (github.com/dnrgus/ctf-knowledge-base) ships ``tools/retrieve.py``, a
lexical retriever with no LLM in the loop. This module runs it on the host, not
in the solver sandbox, so the sandbox never sees the KB tree.

Level B (name + overview + indicators + requires) is used because the KB's own
benchmark (reports/retrieval-benchmark.md) measured it best for top-k. Levels D
and E add blocker and chain scoring, and their wrong-route rate rose from 16.7%
to 53.3%. Results are hints to check against the challenge's own evidence, not facts.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

KB_LEVEL = "B"
KB_TIMEOUT_S = 60.0
KB_OUTPUT_CHARS = 4000
KB_EVIDENCE_CHARS = 2000
KB_HINT_HEADER = (
    "KB hints (unverified: check each technique's preconditions against this challenge "
    "before using it):"
)


async def search_kb(kb_dir: str, evidence: str, category: str = "", top: int = 5) -> str:
    if not kb_dir:
        return "KB is not configured. The operator can enable it with --kb-dir."
    retriever = Path(kb_dir).expanduser() / "tools" / "retrieve.py"
    if not retriever.is_file():
        return f"KB retriever not found at {retriever}."
    evidence = evidence.strip()[:KB_EVIDENCE_CHARS]
    if not evidence:
        return "kb_search needs a non-empty evidence description."
    argv = [sys.executable, str(retriever), "--evidence", evidence, "--level", KB_LEVEL, "--top", str(top)]
    if category:
        argv += ["--category", category]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(retriever.parent.parent),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=KB_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        return "kb_search timed out."
    except OSError as e:
        return f"kb_search could not run: {e}"
    if proc.returncode != 0:
        return f"kb_search failed (exit {proc.returncode}): {err.decode(errors='replace')[-300:]}"
    body = out.decode(errors="replace").strip()
    if not body:
        return "kb_search: no matching techniques."
    return f"{KB_HINT_HEADER}\n{body[:KB_OUTPUT_CHARS]}"
