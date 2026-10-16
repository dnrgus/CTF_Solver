"""Behavioral tests for ChallengeSwarm using the mock provider + fake sandbox."""

from __future__ import annotations

import asyncio

from conftest import FakeSandbox, MockProvider, resp, tool

from ctf_solver.config import Settings
from ctf_solver.events import EventBus, SolverEvent
from ctf_solver.session_state import SessionState, SolverState
from ctf_solver.solver.solver_base import ResultStatus
from ctf_solver.solver.swarm import (
    SUBMISSION_COOLDOWNS,
    ChallengeSwarm,
    SolverInstance,
)


def _settings(tmp_path, **kw) -> Settings:
    base = {
        "no_docker": True,
        "no_coordinator": True,
        "log_dir": str(tmp_path),
        "output_dir": str(tmp_path),
        "timeout": 30,
        "max_steps": 50,
        "max_cost": 100.0,
        "accept_regex_flag_as_verified": True,
    }
    base.update(kw)
    return Settings(**base)


def _swarm(tmp_path, connection_info: str = "", **settings_kw) -> ChallengeSwarm:
    # connection_info is a ChallengeSwarm field, not a Settings one — pulled out
    # explicitly so passing it here doesn't silently vanish into Settings'
    # extra="ignore" (it used to: there was no ChallengeSwarm field for it at all).
    return ChallengeSwarm(
        challenge_dir=str(tmp_path),
        challenge_name="test",
        description="solve me",
        category="misc",
        settings=_settings(tmp_path, **settings_kw),
        event_bus=EventBus(),
        connection_info=connection_info,
    )


def _bash_turn(command: str, text: str = "") -> object:
    """A non-terminal response that issues one bash tool call."""
    return resp(text=text, tool_calls=[tool("bash", command=command)])


def _drain(bus: EventBus, queue: asyncio.Queue[SolverEvent]) -> list[SolverEvent]:
    """Pull every queued event without blocking, then unsubscribe."""
    events: list[SolverEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    bus.unsubscribe(queue)
    return events


async def test_swarm_solves_via_submit_flag(tmp_path, patch_backends):
    provider = MockProvider(
        {"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{win}")])]}
    )
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)

    result = await swarm.run()

    assert result is not None
    assert result.status == ResultStatus.SOLVED
    assert result.flag == "FLAG{win}"
    assert swarm.winner_id == "claude-0"


async def test_regex_text_flag_is_accepted_as_solved(tmp_path, patch_backends):
    """A bare flag in assistant text solves when regex flags are trusted."""
    provider = MockProvider({"claude-0": [resp(text="I found it: FLAG{from_text}")]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, accept_regex_flag_as_verified=True)

    result = await swarm.run()

    assert result is not None
    assert result.status == ResultStatus.SOLVED
    assert result.flag == "FLAG{from_text}"
    assert swarm.confirmed_flag == "FLAG{from_text}"
    assert swarm.winner_id == "claude-0"


async def test_first_flag_wins_and_loser_is_cancelled(tmp_path, patch_backends):
    """Winner locks the flag on turn 1; a long-running loser cannot override it."""
    provider = MockProvider(
        {
            "claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{winner}")])],
            # Loser keeps issuing distinct bash calls; it never produces a flag.
            "codex-0": [_bash_turn(f"work-{i}") for i in range(40)],
        }
    )
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, codex_count=1)

    result = await swarm.run()

    assert result is not None
    assert result.status == ResultStatus.SOLVED
    assert result.flag == "FLAG{winner}"
    assert swarm.winner_id == "claude-0"
    # The flag the loser could never set stays the winner's.
    assert swarm.confirmed_flag == "FLAG{winner}"
    assert swarm.solvers["codex-0"].flag != "FLAG{winner}" or swarm.solvers["codex-0"].confirmed is False


async def test_duplicate_flag_is_deduped(tmp_path, patch_backends):
    """The same candidate submitted twice is recorded once and reports 'Already tried'."""
    provider = MockProvider(
        {
            "claude-0": [
                resp(tool_calls=[tool("submit_flag", flag="FLAG{dup}")]),
                resp(tool_calls=[tool("submit_flag", flag="FLAG{dup}")]),
            ]
        }
    )
    patch_backends(provider)
    # Keep accept_regex off so the first submit doesn't end the run as SOLVED.
    swarm = _swarm(tmp_path, claude_count=1, accept_regex_flag_as_verified=False)

    await swarm.run()

    assert swarm._submitted_flags == {"FLAG{dup}"}

    # And the dedup branch returns the "Already tried" message directly.
    inst = swarm.solvers["claude-0"]
    msg, is_valid = await swarm._handle_flag_candidate(inst, "FLAG{dup}")
    assert is_valid is False
    assert "Already tried" in msg


async def test_cooldown_blocks_second_distinct_submission(tmp_path, patch_backends):
    """Two distinct wrong flags in quick succession: the 2nd hits the escalating cooldown."""
    provider = MockProvider({"claude-0": []})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, accept_regex_flag_as_verified=False)
    inst = SolverInstance(
        solver_id="claude-0",
        provider_name="claude",
        model_spec="m",
        sandbox=FakeSandbox(),
    )

    # First distinct pattern-matching flag: not verified -> counts as a wrong submit.
    msg1, valid1 = await swarm._handle_flag_candidate(inst, "FLAG{first}")
    assert valid1 is False
    assert inst._wrong_submit_count == 1

    # Second distinct flag immediately after: cooldown index 1 == 30s window is active.
    msg2, valid2 = await swarm._handle_flag_candidate(inst, "FLAG{second}")
    assert valid2 is False
    assert "COOLDOWN" in msg2
    # The blocked attempt does not itself increment the wrong-submit counter.
    assert inst._wrong_submit_count == 1

    # Sanity-check the cooldown table wiring the swarm relies on.
    assert SUBMISSION_COOLDOWNS[1] == 30
    cooled, _ = swarm._check_cooldown(inst)
    assert cooled is False


async def test_max_steps_caps_the_loop(tmp_path, patch_backends):
    """A solver that only ever issues bash calls is stopped at max_steps -> FAILED."""
    provider = MockProvider(
        {"claude-0": [_bash_turn(f"step-{i}") for i in range(20)]}
    )
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_steps=6)

    result = await swarm.run()

    # No winner: run() returns None, but the solver result is recorded.
    assert result is None
    assert swarm._results["claude-0"] == ResultStatus.FAILED
    assert swarm.solvers["claude-0"].step_count == 6


async def test_max_cost_stops_before_solving(tmp_path, patch_backends):
    """A tiny budget trips is_over_budget after the first usage -> FAILED, never SOLVED."""
    provider = MockProvider(
        {"claude-0": [_bash_turn(f"spend-{i}") for i in range(20)]}
    )
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_cost=1e-9)

    result = await swarm.run()

    assert result is None
    assert swarm._results["claude-0"] == ResultStatus.FAILED
    # Budget tripped after exactly one priced turn.
    assert swarm.cost_tracker.total_cost_usd > 0
    assert swarm.solvers["claude-0"].step_count <= 2


async def test_provider_error_returns_error_and_trips_circuit_breaker(
    tmp_path, patch_backends, monkeypatch
):
    """When every retry of send() raises, the solver ends ERROR and the breaker counts it."""
    import ctf_solver.solver.swarm as swarm_mod

    def _boom():
        raise RuntimeError("boom")

    # Repeat the throwing entry past the retry budget so retry_with_backoff re-raises.
    provider = MockProvider({"claude-0": [_boom, _boom, _boom, _boom]})
    patch_backends(provider)

    # Keep the retry backoff from actually sleeping (3 retries would burst ~3s).
    async def _instant_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(swarm_mod.asyncio, "sleep", _instant_sleep)

    swarm = _swarm(tmp_path, claude_count=1)
    result = await swarm.run()

    assert result is None
    assert swarm._results["claude-0"] == ResultStatus.ERROR
    cb = swarm._circuit_breakers["claude"]
    assert cb._consecutive_failures >= 1


async def test_repeated_identical_tool_call_triggers_loop_detection(tmp_path, patch_backends):
    """The same bash command run over and over is caught by the loop detector."""
    provider = MockProvider(
        {"claude-0": [resp(tool_calls=[tool("bash", command="x")]) for _ in range(8)]}
    )
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)

    result = await swarm.run()

    assert result is None
    assert swarm._results["claude-0"] == ResultStatus.LOOP_DETECTED


async def test_insight_sharing_posts_findings_and_publishes_event(tmp_path, patch_backends):
    """Crossing step 5 with findings posts a 'finding' and emits an insight_shared event."""
    # 7 turns each with text (-> findings) and a distinct bash call (avoids loop break).
    script = [_bash_turn(f"cmd-{i}", text=f"finding number {i}") for i in range(7)]
    provider = MockProvider({"claude-0": list(script), "codex-0": list(script)})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, codex_count=1, max_steps=30)

    queue = swarm.event_bus.subscribe()
    await swarm.run()
    events = _drain(swarm.event_bus, queue)

    findings = [m for m in swarm.message_bus.findings if m.category == "finding"]
    assert findings, "expected at least one shared 'finding' on the message bus"
    assert any(m.solver_id in {"claude-0", "codex-0"} for m in findings)

    shared = [e for e in events if e.type == "insight_shared"]
    assert shared, "expected an insight_shared event to be published"


async def test_kill_cancels_running_solver(tmp_path, patch_backends):
    """A solver parked inside a tool call returns CANCELLED when its task is cancelled."""
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("bash", command="hang")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)
    swarm._circuit_breakers = swarm._create_circuit_breakers()

    entered = asyncio.Event()
    never = asyncio.Event()

    async def _blocking_exec(command: str, timeout_s: int = 300):
        entered.set()
        await never.wait()  # parks here until the task is cancelled

    inst = SolverInstance(
        solver_id="claude-0",
        provider_name="claude",
        model_spec="m",
        sandbox=FakeSandbox(),
    )
    inst.sandbox.exec = _blocking_exec  # type: ignore[method-assign]

    task = asyncio.create_task(swarm._run_solver(inst))
    await entered.wait()  # deterministic: we are parked inside the tool
    swarm.kill()  # sets cancel_event (mirrors the production kill path)
    task.cancel()

    result = await task
    assert result is not None
    assert result.status == ResultStatus.CANCELLED


async def test_resume_skips_already_solved_solver(tmp_path, patch_backends):
    """A solver SOLVED in a prior session is not re-run and its flag carries forward."""
    provider = MockProvider(
        {"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{new}")])]}
    )
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)
    swarm.resume_state = SessionState(
        challenge_name="test",
        challenge_dir=str(tmp_path),
        description="solve me",
        category="misc",
        confirmed_flag="FLAG{old}",
        winner_id="claude-0",
        solvers=[
            SolverState(
                solver_id="claude-0",
                provider="claude",
                status="solved",
                flag="FLAG{old}",
            )
        ],
    )

    result = await swarm.run()

    # claude-0 was skipped, so no session was ever created for it.
    assert "claude-0" not in provider.sessions
    assert swarm._results["claude-0"] == ResultStatus.SOLVED
    # Prior flag/winner carried forward; the would-be new flag never ran.
    assert swarm.solvers["claude-0"].flag == "FLAG{old}"
    assert swarm.confirmed_flag == "FLAG{old}"
    assert swarm.winner_id == "claude-0"
    # Only solver was skipped -> no tasks scheduled -> run() returns None.
    assert result is None


async def test_regex_text_mention_does_not_penalize_submit(tmp_path, patch_backends):
    """A flag merely *mentioned* in reasoning text must not trip the escalating
    cooldown that would block the model's next deliberate submit_flag."""
    provider = MockProvider({"claude-0": [resp(text="I suspect the flag is FLAG{maybe}")]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, accept_regex_flag_as_verified=False)

    await swarm.run()

    inst = swarm.solvers["claude-0"]
    # The candidate was recorded (deduped) but carried no wrong-submit penalty.
    assert "FLAG{maybe}" in swarm._submitted_flags
    assert inst._wrong_submit_count == 0
    cooled, _ = swarm._check_cooldown(inst)
    assert cooled is True


async def test_connection_info_reaches_system_prompt(tmp_path, patch_backends):
    """ChallengeSwarm.connection_info flows into ChallengeMeta -> build_prompt,
    so the service-connection prompt section (ported from verialabs) is
    actually reachable — it used to be dropped on the floor before this
    (the field didn't even exist on ChallengeSwarm)."""
    provider = MockProvider({"claude-0": [resp(text="no flag yet")]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, connection_info="nc 1.2.3.4 1337", claude_count=1)

    await swarm.run()

    prompt = provider.sessions["claude-0"].system_prompt
    assert "nc 1.2.3.4 1337" in prompt
    assert "FIRST ACTION REQUIRED" in prompt
    assert "## Service Connection" in prompt


async def test_empty_connection_info_skips_service_section(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(text="no flag yet")]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)  # connection_info defaults to ""

    await swarm.run()

    prompt = provider.sessions["claude-0"].system_prompt
    assert "FIRST ACTION REQUIRED" not in prompt
    assert "## Service Connection" not in prompt
