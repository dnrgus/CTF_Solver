"""Logs screen — JSONL-style trace viewer, filterable by solver id."""

from __future__ import annotations

import asyncio
import json
import logging

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, RichLog, Select

from ctf_solver.events import SolverEvent

logger = logging.getLogger(__name__)

ALL_SOLVERS = "__all__"


class LogsScreen(Screen):
    """Detailed event-trace viewer.

    Subscribes to the event bus and renders each event as a compact JSON line
    (mirroring the on-disk JSONL trace format). The dropdown lets the operator
    narrow the feed to a single solver id; ``__all__`` shows everything.
    """

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    DEFAULT_CSS = """
    LogsScreen #log-controls {
        height: 3;
        dock: top;
    }
    LogsScreen #solver-filter {
        width: 40;
    }
    LogsScreen #filter-label {
        padding: 1 1 0 1;
    }
    LogsScreen RichLog {
        height: 1fr;
        border: solid blue;
    }
    """

    def __init__(self, event_bus, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus
        self._log_task: asyncio.Task | None = None
        # Keep the raw events so we can re-render when the filter changes.
        self._records: list[SolverEvent] = []
        self._solver_ids: list[str] = []
        self._filter: str = ALL_SOLVERS

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="log-controls"):
            yield Label("Filter solver:", id="filter-label")
            yield Select(
                [("all solvers", ALL_SOLVERS)],
                value=ALL_SOLVERS,
                allow_blank=False,
                id="solver-filter",
            )
        yield RichLog(id="logs-detail", markup=False, wrap=True, max_lines=2000)
        yield Footer()

    async def on_mount(self) -> None:
        self._log_task = asyncio.create_task(self._event_loop())

    async def _event_loop(self) -> None:
        queue = self.event_bus.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    self.add_record(event)
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
        finally:
            self.event_bus.unsubscribe(queue)

    # ----- record handling ------------------------------------------------------

    def add_record(self, event: SolverEvent) -> None:
        """Store an event and append it to the log if it passes the active filter."""
        self._records.append(event)
        if event.solver_id not in self._solver_ids:
            self._solver_ids.append(event.solver_id)
            self._refresh_filter_options()
        if self._passes_filter(event):
            self._write_line(event)

    def _passes_filter(self, event: SolverEvent) -> bool:
        return self._filter == ALL_SOLVERS or event.solver_id == self._filter

    def _format_line(self, event: SolverEvent) -> str:
        record = {"solver": event.solver_id, "type": event.type, "data": event.data}
        try:
            return json.dumps(record, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return f'{{"solver": "{event.solver_id}", "type": "{event.type}", "data": "<unserializable>"}}'

    def _write_line(self, event: SolverEvent) -> None:
        try:
            self.query_one("#logs-detail", RichLog).write(self._format_line(event))
        except Exception:
            logger.exception("Failed to write log line")

    def _refresh_filter_options(self) -> None:
        options = [("all solvers", ALL_SOLVERS)] + [(sid, sid) for sid in self._solver_ids]
        try:
            select = self.query_one("#solver-filter", Select)
            select.set_options(options)
            select.value = self._filter
        except Exception:
            logger.exception("Failed to refresh solver filter options")

    def _rerender(self) -> None:
        try:
            log = self.query_one("#logs-detail", RichLog)
        except Exception:
            logger.exception("Failed to query log widget for re-render")
            return
        log.clear()
        for event in self._records:
            if self._passes_filter(event):
                log.write(self._format_line(event))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "solver-filter":
            return
        self._filter = ALL_SOLVERS if event.value is Select.BLANK else str(event.value)
        self._rerender()

    def on_unmount(self) -> None:
        if getattr(self, "_log_task", None):
            self._log_task.cancel()
