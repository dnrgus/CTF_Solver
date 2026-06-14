"""Coordinator guidance display — shows the coordinator's status and latest guidance."""

from textual.widgets import Static


class CoordinatorView(Static):
    DEFAULT_CSS = """
    CoordinatorView {
        height: auto;
        max-height: 8;
        padding: 1;
        border: solid magenta;
        overflow-y: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__("Coordinator: idle")
        self._messages: list[str] = []
        self._status: str = "idle"

    def add_guidance(self, message: str) -> None:
        self._messages.append(message)
        if len(self._messages) > 20:
            self._messages = self._messages[-20:]
        # Receiving guidance implies the coordinator is doing work.
        self._status = "active"
        self._render_view()

    def set_status(self, status: str) -> None:
        """Update the coordinator's activity status (e.g. ``active``/``idle``)."""
        self._status = status
        self._render_view()

    def _render_view(self) -> None:
        header = f"Coordinator: {self._status}"
        if self._messages:
            body = "\n".join(self._messages[-5:])
            self.update(f"{header}\nGuidance:\n{body}")
        else:
            self.update(header)
