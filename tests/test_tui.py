"""Tests for the Textual TUI — widgets, app event handling, and the logs screen.

These run headless via Textual's ``run_test()`` pilot (no real terminal) and
plain unit tests of widget update methods. ``asyncio_mode = "auto"`` (pyproject)
means the ``async def`` tests run without an explicit marker.
"""

from __future__ import annotations

from ctf_solver.events import EventBus, SolverEvent
from ctf_solver.tui.app import CTFApp
from ctf_solver.tui.screens.logs import ALL_SOLVERS, LogsScreen
from ctf_solver.tui.widgets.coordinator_view import CoordinatorView
from ctf_solver.tui.widgets.cost_bar import CostBar
from ctf_solver.tui.widgets.input_bar import HintInputBar
from ctf_solver.tui.widgets.message_log import MessageLog
from ctf_solver.tui.widgets.solver_panel import SolverPanel

# --------------------------------------------------------------------------- #
# Widget unit tests (no terminal required)
# --------------------------------------------------------------------------- #


def test_solver_panel_lifecycle():
    panel = SolverPanel()
    assert "No solvers running yet." in str(panel.render())
    panel.add_solver("claude-0", {"provider": "claude"})
    panel.update_solver("claude-0", steps=3, cost=1.25, last_action="bash")
    rendered = str(panel.render())
    assert "claude-0" in rendered
    assert "$1.25" in rendered
    assert "3steps" in rendered
    panel.mark_done("claude-0")
    # Updating an unknown solver must not raise.
    panel.update_solver("ghost", steps=1)


def test_message_log_caps_and_appends():
    log = MessageLog()
    for i in range(150):
        log.log_message(f"msg-{i}")
    # Internal buffer is capped at 100.
    assert len(log._messages) == 100
    assert log._messages[-1] == "msg-149"
    assert "msg-149" in str(log.render())


def test_cost_bar_update_and_zero_max():
    bar = CostBar()
    bar.update_cost(2.5, 10.0)
    rendered = str(bar.render())
    assert "$2.50" in rendered
    assert "25%" in rendered
    # max_cost == 0 must not divide-by-zero.
    bar.update_cost(1.0, 0.0)
    assert "0%" in str(bar.render())


def test_coordinator_view_status_and_guidance():
    cv = CoordinatorView()
    assert "idle" in str(cv.render())
    # set_status is wired and reflected even with no guidance yet.
    cv.set_status("active")
    assert "active" in str(cv.render())
    cv.add_guidance("try a buffer overflow")
    rendered = str(cv.render())
    assert "try a buffer overflow" in rendered
    # Receiving guidance flips status to active.
    assert "active" in rendered


def test_app_total_cost_sums_per_solver():
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="demo", max_cost=10.0)
    app._solver_costs["claude-0"] = 1.5
    app._solver_costs["codex-0"] = 2.0
    assert app.total_cost == 3.5


def test_app_status_text_includes_all_fields():
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="my-chall", max_cost=7.0, category="web")
    text = app._status_text()
    assert "my-chall" in text
    assert "web" in text
    assert "Elapsed" in text
    assert "$0.00 / $7.00" in text


def test_app_constructor_backward_compatible_without_category():
    """CLI instantiates CTFApp without ``category`` — must still work."""
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="x", max_cost=10.0)
    assert app.category is None
    assert "Category: ?" in app._status_text()


# --------------------------------------------------------------------------- #
# App integration tests via the pilot harness (headless)
# --------------------------------------------------------------------------- #


async def test_app_handles_events_without_crashing():
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="demo", max_cost=10.0, category="pwn")
    async with app.run_test() as pilot:
        events = [
            SolverEvent(type="solver_started", solver_id="claude-0", data={"provider": "claude"}),
            SolverEvent(type="tool_call", solver_id="claude-0", data={"tool": "bash"}),
            SolverEvent(type="cost_update", solver_id="claude-0", data={"cost": 1.5}),
            SolverEvent(type="cost_update", solver_id="codex-0", data={"cost": 2.0}),
            SolverEvent(type="coordinator_guidance", solver_id="coordinator", data={"guidance": "look at headers"}),
            SolverEvent(type="flag_found", solver_id="claude-0", data={"flag": "CTF{x}"}),
            SolverEvent(type="solver_done", solver_id="claude-0", data={}),
            # Intentionally malformed data — must be logged, not crash.
            SolverEvent(type="cost_update", solver_id="bad", data={}),
        ]
        for ev in events:
            bus.publish(ev)
        await pilot.pause()

        # Running total = sum of latest per-solver costs (1.5 + 2.0 + 0.0).
        assert app.total_cost == 3.5
        cost_bar = app.query_one(CostBar)
        assert "$3.50" in str(cost_bar.render())

        panel = app.query_one(SolverPanel)
        assert "claude-0" in str(panel.render())

        coord = app.query_one(CoordinatorView)
        assert "look at headers" in str(coord.render())


async def test_app_toggle_actions_change_visibility():
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="demo", max_cost=10.0)
    async with app.run_test() as pilot:
        panel = app.query_one(SolverPanel)
        cost_bar = app.query_one(CostBar)
        assert panel.display is True
        assert cost_bar.display is True

        # Invoke the bound actions directly — pressing the literal "s"/"c" keys
        # would be swallowed by the focused HintInputBar (an Input).
        app.action_toggle_status()
        await pilot.pause()
        assert panel.display is False
        app.action_toggle_status()
        await pilot.pause()
        assert panel.display is True

        app.action_toggle_cost()
        await pilot.pause()
        assert cost_bar.display is False
        app.action_toggle_cost()
        await pilot.pause()
        assert cost_bar.display is True


async def test_app_hint_submission_publishes_event():
    bus = EventBus()
    queue = bus.subscribe()
    app = CTFApp(event_bus=bus, challenge_name="demo", max_cost=10.0)
    async with app.run_test() as pilot:
        hint = app.query_one(HintInputBar)
        hint.value = "check the cookies"
        await hint.action_submit()
        await pilot.pause()
    # Drain the queue looking for our operator hint.
    found = False
    while not queue.empty():
        ev = queue.get_nowait()
        if ev.type == "user_hint" and ev.data.get("message") == "check the cookies":
            found = True
    assert found


async def test_app_status_line_present_in_dom():
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="demo", max_cost=10.0, category="crypto")
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Static

        status = app.query_one("#status-line", Static)
        text = str(status.render())
        assert "demo" in text
        assert "crypto" in text


# --------------------------------------------------------------------------- #
# Logs screen tests
# --------------------------------------------------------------------------- #


async def test_logs_screen_filters_by_solver():
    bus = EventBus()
    app = CTFApp(event_bus=bus, challenge_name="demo", max_cost=10.0)
    async with app.run_test() as pilot:
        app.push_screen(LogsScreen(bus))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LogsScreen)

        screen.add_record(SolverEvent(type="tool_call", solver_id="claude-0", data={"tool": "bash"}))
        screen.add_record(SolverEvent(type="tool_call", solver_id="codex-0", data={"tool": "python"}))
        await pilot.pause()

        # Both solver ids are tracked and offered as filter options.
        assert "claude-0" in screen._solver_ids
        assert "codex-0" in screen._solver_ids
        assert len(screen._records) == 2

        # Default filter shows everything.
        assert screen._filter == ALL_SOLVERS

        # Narrowing to one solver only keeps that solver's records on re-render.
        screen._filter = "claude-0"
        screen._rerender()
        await pilot.pause()
        kept = [r for r in screen._records if screen._passes_filter(r)]
        assert len(kept) == 1
        assert kept[0].solver_id == "claude-0"


def test_logs_format_line_is_json():
    import json

    screen = LogsScreen(EventBus())
    line = screen._format_line(SolverEvent(type="tool_call", solver_id="claude-0", data={"tool": "bash"}))
    parsed = json.loads(line)
    assert parsed["solver"] == "claude-0"
    assert parsed["type"] == "tool_call"
    assert parsed["data"]["tool"] == "bash"


def test_logs_format_line_handles_unserializable():
    screen = LogsScreen(EventBus())
    # A non-JSON-serializable object in data should fall back gracefully.
    line = screen._format_line(SolverEvent(type="tool_result", solver_id="s", data={"obj": object()}))
    assert "claude" not in line
    assert '"solver": "s"' in line or '"solver":"s"' in line
