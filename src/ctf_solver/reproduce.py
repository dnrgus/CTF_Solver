"""Independent reproduction of a candidate flag from the solver's solve.py.

The script is copied into a brand-new sandbox (its own workspace, no state
from the solving agent) and run from scratch. The candidate counts as
LOCALLY_REPRODUCED only if:

- the flag literal does not appear in the script source (a script that prints
  a hardcoded string proves nothing), and
- the script exits 0 and its stdout contains the flag.

A remote-dependent script needs the target to be reachable from the fresh
sandbox. If it is not, the run ends as FAILED with that reason, not as a
success.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ctf_solver.verification import FlagStatus

SOLVE_SCRIPT_PATH = "/challenge/workspace/solve.py"
REPRODUCE_TIMEOUT_S = 240
_NOTE_TAIL_CHARS = 600


REMOTE_CAVEAT = (
    " | the challenge has a remote target and reproduction does not check its reachability from "
    "the fresh sandbox, so this failure may be a network problem rather than a wrong solve"
)


async def reproduce_candidate(
    flag: str,
    solve_source: str | None,
    make_sandbox: Callable[[], Awaitable[Any]],
    remote_target: str = "",
) -> tuple[FlagStatus, str]:
    """Re-run ``solve_source`` in a fresh sandbox and check it prints ``flag``.

    ``make_sandbox`` returns a new, not-yet-started sandbox. This function starts
    it and always stops it. ``remote_target`` only changes the note on failure:
    a remote-dependent solve can fail for reasons this check cannot see.
    """
    if not solve_source or not solve_source.strip():
        return FlagStatus.CANDIDATE, "no solve.py yet — write one that reproduces the flag from scratch"
    if flag in solve_source:
        return FlagStatus.FAILED, "solve.py contains the flag literal — hardcoded output is not a reproduction"

    sandbox = await make_sandbox()
    try:
        await sandbox.start()
        await sandbox.write_file(SOLVE_SCRIPT_PATH, solve_source)
        result = await sandbox.exec(
            f"cd /challenge/workspace && timeout {REPRODUCE_TIMEOUT_S} python3 {SOLVE_SCRIPT_PATH}",
            timeout_s=REPRODUCE_TIMEOUT_S + 30,
        )
    except Exception as e:  # sandbox start/exec failure is not a verdict on the flag
        return FlagStatus.UNKNOWN, f"fresh sandbox run failed: {type(e).__name__}: {e}"
    finally:
        try:
            await sandbox.stop()
        except Exception:
            pass

    output = (result.stdout or "") + (("\n[stderr]\n" + result.stderr) if result.stderr else "")
    tail = output.strip()[-_NOTE_TAIL_CHARS:]
    if result.exit_code == 0 and flag in (result.stdout or ""):
        return FlagStatus.LOCALLY_REPRODUCED, f"fresh sandbox reproduced the flag (exit 0): {tail}"
    caveat = REMOTE_CAVEAT if remote_target else ""
    return FlagStatus.FAILED, (
        f"fresh sandbox run did not print the flag (exit {result.exit_code}): {tail}{caveat}"
    )
