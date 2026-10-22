"""Tests for CoordinatorAgent.

The coordinator reads solver traces, injects strategic guidance onto the shared
message bus, relays operator hints from the EventBus, and records its own token
usage under the agent id ``"coordinator"``.

These tests drive it deterministically with no network: the provider is swapped
for the ``MockProvider``/``MockSession`` from conftest by monkeypatching
``ctf_solver.solver.coordinator.get_provider``, and the 30s coordination
interval is monkeypatched to 0 so a tick fires immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from conftest import MockProvider, resp

import ctf_solver.solver.coordinator as coordinator_mod
from ctf_solver.collaboration.message_bus import ChallengeMessageBus
from ctf_solver.config import Settings
from ctf_solver.events import EventBus, SolverEvent
from ctf_solver.providers.base import TokenUsage
from ctf_solver.solver.coordinator import CoordinatorAgent

# --- minimal swarm stand-in -------------------------------------------------
# The coordination loop reads exactly three things off the swarm:
#   * ``get_status()["solvers"]`` -> {solver_id: {"steps": int, ...}}
#   * ``solvers.get(solver_id)``  -> object exposing ``.tracer.path``
#   * ``_model_spec_for(provider_name)`` -> model name string for cost tracking


@dataclass
class _FakeTracer:
    path: str


@dataclass
class _FakeSolver:
    tracer: _FakeTracer | None


@dataclass
class _FakeSwarm:
    solvers: dict = field(default_factory=dict)
    model_spec: str = "mock-model"

    def get_status(self) -> dict:
        return {
            "solvers": {sid: {"steps": 3} for sid in self.solvers},
        }

    def _model_spec_for(self, provider_name: str) -> str:
        return self.model_spec


def _make_swarm_with_trace(tmp_path, lines: list[str]) -> _FakeSwarm:
    """Build a fake swarm whose single solver has a populated trace file."""
    trace = tmp_path / "solver.jsonl"
    trace.write_text("\n".join(lines))
    solver = _FakeSolver(tracer=_FakeTracer(path=str(trace)))
    return _FakeSwarm(solvers={"mock-0": solver})


def _make_coordinator(monkeypatch, *, swarm=None, script=None) -> CoordinatorAgent:
    """Construct a coordinator wired to a MockProvider (no network).

    ``script`` is the list of responses the coordinator's session replays; it is
    registered under solver id ``"coordinator"`` since that is the id the
    coordinator uses when it opens its session.
    """
    provider = MockProvider(scripts={"coordinator": list(script or [])})
    monkeypatch.setattr(coordinator_mod, "get_provider", lambda name: provider)
    return CoordinatorAgent(
        provider_name="mock",
        settings=Settings(),
        event_bus=EventBus(),
        cost_tracker=coordinator_mod.CostTracker(),
        message_bus=ChallengeMessageBus(),
        swarm=swarm,
    )


async def _pump(times: int = 5) -> None:
    """Yield control to the event loop so background tasks make progress."""
    import asyncio

    for _ in range(times):
        await asyncio.sleep(0)


# --- _summarize_trace -------------------------------------------------------


def test_coordinator_summarize_trace():
    agent = CoordinatorAgent.__new__(CoordinatorAgent)
    lines = [
        json.dumps({"type": "tool_call", "tool": "bash", "step": 1, "args": {"command": "ls"}}),
        json.dumps({"type": "tool_result", "tool": "bash", "step": 1, "result": "file.txt"}),
    ]
    summary = agent._summarize_trace(lines)
    assert "bash" in summary
    assert "Step 1" in summary


def test_coordinator_summarize_empty():
    agent = CoordinatorAgent.__new__(CoordinatorAgent)
    summary = agent._summarize_trace([])
    assert "No recent" in summary


def test_coordinator_summarize_finish_and_error_lines():
    agent = CoordinatorAgent.__new__(CoordinatorAgent)
    lines = [
        json.dumps({"type": "finish", "flag": "FLAG{x}"}),
        json.dumps({"type": "error", "message": "boom"}),
        json.dumps({"type": "bump", "reason": "nudge"}),
    ]
    summary = agent._summarize_trace(lines)
    assert "finish" in summary
    assert "error" in summary
    assert "bump" in summary


def test_coordinator_summarize_malformed_line_falls_back_to_raw():
    agent = CoordinatorAgent.__new__(CoordinatorAgent)
    # Not JSON -> the parser swallows the error and keeps a truncated raw line.
    lines = ["this is not json at all", json.dumps({"type": "tool_call", "tool": "nmap", "step": 7})]
    summary = agent._summarize_trace(lines)
    assert "this is not json" in summary
    assert "nmap" in summary
    assert "Step 7" in summary


def test_coordinator_summarize_unknown_type_is_dropped():
    agent = CoordinatorAgent.__new__(CoordinatorAgent)
    # A well-formed line with a type the summarizer ignores produces no parts,
    # so the summary collapses to the "no activity" sentinel.
    lines = [json.dumps({"type": "heartbeat", "step": 1})]
    summary = agent._summarize_trace(lines)
    assert summary == "No recent activity."


# --- hint relay -------------------------------------------------------------


@pytest.mark.asyncio
async def test_relay_hints_broadcasts_operator_hint_to_message_bus(monkeypatch):
    coordinator = _make_coordinator(monkeypatch)
    await coordinator.start()
    try:
        coordinator.event_bus.publish(
            SolverEvent(type="user_hint", solver_id="operator", data={"message": "check headers"})
        )
        # _relay_hints polls the subscription with a 1s timeout; a few loop
        # turns are enough for it to drain the queued event.
        await _pump(20)

        unread = await coordinator.message_bus.check("someone")
        assert any("check headers" in m.content for m in unread)
        assert any("Operator hint" in m.content for m in unread)
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_relay_hints_ignores_non_hint_events(monkeypatch):
    coordinator = _make_coordinator(monkeypatch)
    await coordinator.start()
    try:
        # Wrong event type and an empty-message hint must not be broadcast.
        coordinator.event_bus.publish(
            SolverEvent(type="tool_call", solver_id="mock-0", data={"message": "ignore me"})
        )
        coordinator.event_bus.publish(
            SolverEvent(type="user_hint", solver_id="operator", data={"message": ""})
        )
        await _pump(20)

        assert coordinator.message_bus.findings == []
    finally:
        await coordinator.stop()


# --- guidance broadcast -----------------------------------------------------


@pytest.mark.asyncio
async def test_guidance_tick_broadcasts_and_publishes_event(monkeypatch, tmp_path):
    # Drive ONE coordination tick by collapsing the 30s interval to 0 so the
    # loop body runs immediately on the first iteration.
    monkeypatch.setattr(coordinator_mod, "COORDINATION_INTERVAL", 0)
    swarm = _make_swarm_with_trace(
        tmp_path,
        [json.dumps({"type": "tool_call", "tool": "bash", "step": 1})],
    )
    coordinator = _make_coordinator(
        monkeypatch, swarm=swarm, script=[resp(text="Try a buffer overflow")]
    )

    # Capture published events off the same EventBus the coordinator uses.
    sub = coordinator.event_bus.subscribe()

    await coordinator.start()
    try:
        await _pump(30)

        unread = await coordinator.message_bus.check("listener")
        assert any("Try a buffer overflow" in m.content for m in unread)
        assert any(m.solver_id == "coordinator" for m in unread)

        events = []
        while not sub.empty():
            events.append(sub.get_nowait())
        guidance_events = [e for e in events if e.type == "coordinator_guidance"]
        assert guidance_events, "expected a coordinator_guidance event to be published"
        assert "Try a buffer overflow" in guidance_events[0].data["guidance"]
    finally:
        await coordinator.stop()


# --- cost tracking ----------------------------------------------------------


@pytest.mark.asyncio
async def test_guidance_tick_records_coordinator_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(coordinator_mod, "COORDINATION_INTERVAL", 0)
    swarm = _make_swarm_with_trace(
        tmp_path,
        [json.dumps({"type": "tool_call", "tool": "bash", "step": 1})],
    )
    coordinator = _make_coordinator(
        monkeypatch,
        swarm=swarm,
        script=[resp(text="guidance", usage=TokenUsage(input_tokens=42, output_tokens=7))],
    )

    await coordinator.start()
    try:
        await _pump(30)

        by_agent = coordinator.cost_tracker.by_agent
        assert "coordinator" in by_agent
        usage = by_agent["coordinator"]
        # With the interval collapsed to 0 the loop ticks repeatedly while we
        # pump, so totals accumulate. We only assert the coordinator's usage was
        # recorded and includes at least the first scripted send (42 in / 7 out).
        assert usage.input_tokens >= 42
        assert usage.output_tokens >= 7
        # The model name recorded comes from swarm._model_spec_for(...).
        assert usage.model_name == "mock-model"
        assert usage.provider_spec == "mock"
    finally:
        await coordinator.stop()


# --- stop() cleanliness -----------------------------------------------------


@pytest.mark.asyncio
async def test_stop_cancels_tasks_and_is_idempotent(monkeypatch):
    coordinator = _make_coordinator(monkeypatch)
    await coordinator.start()

    assert coordinator._task is not None
    assert coordinator._hint_task is not None
    hint_task = coordinator._hint_task
    loop_task = coordinator._task

    # First stop should cancel both background tasks without raising.
    await coordinator.stop()
    assert hint_task.done()
    assert loop_task.done()
    assert coordinator.session is not None and coordinator.session.closed

    # Second stop must be a no-op, not raise.
    await coordinator.stop()


@pytest.mark.asyncio
async def test_start_disabled_when_provider_config_invalid(monkeypatch):
    # If the provider reports its config is invalid, start() bails before
    # creating a session or any background tasks.
    provider = MockProvider()
    monkeypatch.setattr(provider, "validate_config", lambda config: False)
    monkeypatch.setattr(coordinator_mod, "get_provider", lambda name: provider)
    coordinator = CoordinatorAgent(
        provider_name="mock",
        settings=Settings(),
        event_bus=EventBus(),
        cost_tracker=coordinator_mod.CostTracker(),
        message_bus=ChallengeMessageBus(),
    )

    await coordinator.start()
    try:
        assert coordinator.session is None
        assert coordinator._task is None
        assert coordinator._hint_task is None
    finally:
        # stop() must be safe even though start() short-circuited.
        await coordinator.stop()


# --- --no-coordinator guard -------------------------------------------------


def test_no_coordinator_factory_returns_none():
    from ctf_solver.config import get_coordinator_provider

    s = Settings(claude_count=1, no_coordinator=True)
    assert get_coordinator_provider(s) is None


def test_coordinator_factory_defaults_to_first_active_provider():
    from ctf_solver.config import get_coordinator_provider

    s = Settings(claude_count=1, codex_count=2)
    assert get_coordinator_provider(s) == "claude"


# --------------------------------------------------------------------------- #
# Codex containment: the coordinator's own codex session must be confined to
# a Docker sandbox exactly like a solver's — its tools=[] did NOT disable
# codex's own native agent tooling (confirmed live: a bare, unsandboxed
# coordinator session read real host files). See providers/codex.py's
# containment comments and solver/swarm.py's equivalent per-solver wiring.
# --------------------------------------------------------------------------- #
class _FakeCodexSandbox:
    def __init__(self, workspace_root: str = "", **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self._workspace_root = workspace_root
        self.workspace_dir = ""
        self.container_id = "fake-coordinator-container"

    async def start(self) -> None:
        self.started = True
        self.workspace_dir = self._workspace_root

    async def stop(self) -> None:
        self.stopped = True


async def test_codex_coordinator_gets_its_own_sandbox(monkeypatch, tmp_path):
    provider = MockProvider(scripts={"coordinator": []})
    monkeypatch.setattr(coordinator_mod, "get_provider", lambda name: provider)
    monkeypatch.setattr(coordinator_mod, "locate_codex_native_bin_dir", lambda: "/fake/codex-bin")
    created: list[_FakeCodexSandbox] = []
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    def _fake_sandbox_ctor(**kwargs):
        sb = _FakeCodexSandbox(workspace_root=str(workspace_root), **kwargs)
        created.append(sb)
        return sb

    monkeypatch.setattr(coordinator_mod, "DockerSandbox", _fake_sandbox_ctor)
    monkeypatch.setattr(coordinator_mod.Path, "home", lambda: tmp_path)  # no real ~/.codex/auth.json

    coordinator = CoordinatorAgent(
        provider_name="codex",
        settings=Settings(codex_count=1, no_docker=False, allow_container_credentials=True),
        event_bus=EventBus(),
        cost_tracker=coordinator_mod.CostTracker(),
        message_bus=ChallengeMessageBus(),
    )

    await coordinator.start()
    try:
        assert len(created) == 1
        assert created[0].started is True
        assert created[0].kwargs["extra_ro_binds"] == ["/fake/codex-bin:/opt/codex-bin:ro"]
        assert coordinator.session.sandbox is created[0]
    finally:
        await coordinator.stop()

    assert created[0].stopped is True


async def test_codex_coordinator_no_sandbox_in_no_docker_mode(monkeypatch):
    """--no-docker is already gated behind its own explicit danger prompt
    elsewhere — the coordinator must not try to build a container there."""
    provider = MockProvider(scripts={"coordinator": []})
    monkeypatch.setattr(coordinator_mod, "get_provider", lambda name: provider)
    called = []
    monkeypatch.setattr(
        coordinator_mod, "DockerSandbox", lambda **kw: called.append(1) or _FakeCodexSandbox(**kw)
    )

    coordinator = CoordinatorAgent(
        provider_name="codex",
        settings=Settings(codex_count=1, no_docker=True),
        event_bus=EventBus(),
        cost_tracker=coordinator_mod.CostTracker(),
        message_bus=ChallengeMessageBus(),
    )

    await coordinator.start()
    try:
        assert called == []
        assert coordinator._sandbox is None
    finally:
        await coordinator.stop()


async def test_non_codex_coordinator_never_builds_a_sandbox(monkeypatch):
    provider = MockProvider(scripts={"coordinator": []})
    monkeypatch.setattr(coordinator_mod, "get_provider", lambda name: provider)
    called = []
    monkeypatch.setattr(
        coordinator_mod, "DockerSandbox", lambda **kw: called.append(1) or _FakeCodexSandbox(**kw)
    )

    coordinator = CoordinatorAgent(
        provider_name="mock",
        settings=Settings(claude_count=1, no_docker=False),
        event_bus=EventBus(),
        cost_tracker=coordinator_mod.CostTracker(),
        message_bus=ChallengeMessageBus(),
    )

    await coordinator.start()
    try:
        assert called == []
    finally:
        await coordinator.stop()


def test_codex_session_empty_tools_means_empty_dynamic_tools():
    """The coordinator's tools=[] must actually reach codex as an empty
    dynamicTools list, not the full bash-capable SANDBOX_TOOLS — a separate,
    defense-in-depth fix alongside the sandbox (codex's own native tooling
    isn't gated by dynamicTools, but there is no reason to ALSO hand it ours)."""
    from ctf_solver.providers.codex import CodexSession

    with_tools = CodexSession("s", "p", [object()], {})
    without_tools = CodexSession("s", "p", [], {})
    assert with_tools._tools
    assert not without_tools._tools
