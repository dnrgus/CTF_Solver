"""Flag verification states and the operator-supplied verifier.

A flag moves through these states, and only the last two end a run:

- CANDIDATE: matches the challenge's flag format. Says nothing about correctness.
- LOCALLY_REPRODUCED: a solve.py written by the solver printed the flag when
  re-run in a fresh, clean sandbox, and the flag literal is not in that source.
- REMOTE_CONFIRMED: an authoritative check accepted it (the --verifier-cmd the
  operator configured, e.g. a script that asks the CTF scoreboard).
- FAILED: a check rejected it, or a reproduction did not work.
- UNKNOWN: a check could not run (timeout, verifier outage). Not a rejection.

No verifier and no reproduction means the flag stays CANDIDATE. Nothing is
ever promoted to REMOTE_CONFIRMED without a configured authoritative check.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class FlagStatus(StrEnum):
    CANDIDATE = "candidate"
    LOCALLY_REPRODUCED = "locally_reproduced"
    REMOTE_CONFIRMED = "remote_confirmed"
    FAILED = "failed"
    UNKNOWN = "unknown"


# Only these end the run. A candidate never does.
TERMINAL_OK = frozenset({FlagStatus.LOCALLY_REPRODUCED, FlagStatus.REMOTE_CONFIRMED})

VERIFIER_TIMEOUT_S = 60.0
_NOTE_TAIL_CHARS = 400


@dataclass
class FlagRecord:
    flag: str
    solver_id: str
    step: int
    status: FlagStatus = FlagStatus.CANDIDATE
    # "verifier" when an authoritative check decided the status, "reproduction"
    # for a solve.py re-run, "" while only a candidate.
    source: str = ""
    notes: list[str] = field(default_factory=list)
    # Digest of the solve.py that the last reproduction checked. Resubmitting
    # with the same script would only repeat the same result.
    checked_digest: str = ""

    @property
    def verified(self) -> bool:
        return self.status in TERMINAL_OK

    @property
    def finally_rejected(self) -> bool:
        """A verifier rejection is final. A failed reproduction may be retried
        after the solver fixes its script, so it is not final."""
        return self.status == FlagStatus.FAILED and self.source == "verifier"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class CommandVerifier:
    """Runs an operator-configured check for one candidate flag.

    The command runs on the host, not in the solver sandbox, so it can hold
    credentials the solver must never see. The flag arrives in the CTF_FLAG
    environment variable, never in the command string.

    Exit codes: 0 accepted (REMOTE_CONFIRMED), 1 rejected (FAILED), anything
    else, including a timeout, is UNKNOWN. A verifier outage must not be
    read as a wrong flag.
    """

    def __init__(self, command: str, timeout_s: float = VERIFIER_TIMEOUT_S) -> None:
        if not command.strip():
            msg = "verifier command must not be empty"
            raise ValueError(msg)
        self.command = command
        self.timeout_s = timeout_s

    async def check(self, flag: str) -> tuple[FlagStatus, str]:
        env = {**os.environ, "CTF_FLAG": flag}
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", self.command,
                env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as e:
            return FlagStatus.UNKNOWN, f"verifier could not start: {e}"
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return FlagStatus.UNKNOWN, f"verifier timed out after {self.timeout_s:.0f}s"
        tail = out.decode(errors="replace").strip()[-_NOTE_TAIL_CHARS:]
        if proc.returncode == 0:
            return FlagStatus.REMOTE_CONFIRMED, f"verifier accepted (exit 0): {tail}"
        if proc.returncode == 1:
            return FlagStatus.FAILED, f"verifier rejected (exit 1): {tail}"
        return FlagStatus.UNKNOWN, f"verifier error (exit {proc.returncode}): {tail}"
