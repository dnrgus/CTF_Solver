"""Adaptive scheduling policy: when a second or third agent starts. Pure, no LLM."""

from __future__ import annotations

from ctf_solver.scheduler import AgentView, SchedulerView, decide_spawn


def _view(agents=(), **kw) -> SchedulerView:
    base = {
        "agents": tuple(agents), "planned_left": 2, "spawned": len(tuple(agents)), "max_agents": 3,
        "verified": False, "unverified_candidates": 0, "budget_left_fraction": 1.0,
        "min_budget_fraction": 0.3, "stall_steps": 8,
    }
    base.update(kw)
    return SchedulerView(**base)


def test_running_agent_making_progress_is_not_replaced():
    v = _view([AgentView("a", finished=False, steps_since_novel=2)])
    assert decide_spawn(v) is None


def test_stalled_agent_triggers_a_second_agent():
    v = _view([AgentView("a", finished=False, steps_since_novel=8)], spawned=1, max_agents=2)
    assert decide_spawn(v) == "stalled"


def test_stall_needs_every_running_agent_to_be_stalled():
    v = _view(
        [AgentView("a", finished=False, steps_since_novel=50), AgentView("b", finished=False, steps_since_novel=1)],
        spawned=2, max_agents=3,
    )
    assert decide_spawn(v) is None


def test_refuted_when_all_agents_finished_without_success():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=1, max_agents=2)
    assert decide_spawn(v) == "refuted"


def test_unverified_candidate_with_no_running_agent_triggers_verify():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=1, max_agents=2,
              unverified_candidates=1)
    assert decide_spawn(v) == "verify"


def test_no_spawn_once_verified():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=1, max_agents=3, verified=True)
    assert decide_spawn(v) is None


def test_default_single_agent_cap_never_spawns():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=1, max_agents=1)
    assert decide_spawn(v) is None


def test_cap_never_exceeds_three_even_if_configured_higher():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=3, max_agents=9)
    assert decide_spawn(v) is None


def test_budget_gate_blocks_spawning_when_budget_is_low():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=1, max_agents=3,
              budget_left_fraction=0.2)
    assert decide_spawn(v) is None


def test_nothing_planned_means_nothing_to_spawn():
    v = _view([AgentView("a", finished=True, steps_since_novel=0)], spawned=1, max_agents=3, planned_left=0)
    assert decide_spawn(v) is None
