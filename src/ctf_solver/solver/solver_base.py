"""Solver protocol, result types, and state machine — shared across all backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ctf_solver.providers.base import TokenUsage


class ResultStatus(Enum):
    SOLVED = "solved"  # flag REMOTE_CONFIRMED by the operator-configured verifier
    REPRODUCED = "reproduced"  # flag LOCALLY_REPRODUCED by solve.py in a fresh sandbox
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    LOOP_DETECTED = "loop_detected"
    ERROR = "error"
    QUOTA_ERROR = "quota_error"


class SolverState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHARING = "sharing"
    COORDINATING = "coordinating"
    DONE = "done"


@dataclass
class SolverResult:
    solver_id: str
    status: ResultStatus
    flag: str | None
    steps: int
    duration: float
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    trace_path: Path = field(default_factory=Path)
    findings_summary: str = ""
    error: str | None = None
