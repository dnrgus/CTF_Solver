"""Adaptive agent scheduling: when does a second or third solver get started?

Agent 1 always starts. A later agent starts only for one of these reasons,
all decided from recorded evidence and never from elapsed time:

- "refuted": every running agent has finished without a verified flag (a
  failed hypothesis, a provider error, or exhausted budget), so a replacement
  takes a different approach and gets the refuted-hypothesis handoff.
- "stalled": every running agent has produced no novel output for
  ``stall_steps`` consecutive tool calls.
- "verify": a candidate exists that no one has verified, and no agent is
  still running to check it. The new agent's job is an independent check.

No new agent starts once a flag is verified, once the pool or the agent cap
is used up, or when less than ``min_budget_fraction`` of the cost budget is
left. The default cap of one agent means no concurrency unless the operator
raises ``--max-agents``.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_STALL_STEPS = 8
DEFAULT_MIN_BUDGET_FRACTION = 0.3
MAX_AGENTS = 3


@dataclass(frozen=True)
class AgentView:
    solver_id: str
    finished: bool
    steps_since_novel: int


@dataclass(frozen=True)
class SchedulerView:
    agents: tuple[AgentView, ...]
    planned_left: int
    spawned: int
    max_agents: int
    verified: bool
    unverified_candidates: int
    budget_left_fraction: float
    min_budget_fraction: float = DEFAULT_MIN_BUDGET_FRACTION
    stall_steps: int = DEFAULT_STALL_STEPS


def decide_spawn(view: SchedulerView) -> str | None:
    """The reason to start the next planned agent now, or None to hold off."""
    if view.verified or view.planned_left <= 0:
        return None
    if view.spawned >= min(view.max_agents, MAX_AGENTS):
        return None
    if view.budget_left_fraction < view.min_budget_fraction:
        return None

    running = [a for a in view.agents if not a.finished]
    if not running:
        return "verify" if view.unverified_candidates else "refuted"
    if all(a.steps_since_novel >= view.stall_steps for a in running):
        return "stalled"
    return None
