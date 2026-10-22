from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SolverState:
    solver_id: str
    provider: str
    status: str
    steps: int = 0
    cost_usd: float = 0.0
    flag: str | None = None
    error: str | None = None


@dataclass
class SessionState:
    challenge_name: str
    challenge_dir: str
    description: str
    category: str
    started_at: float = field(default_factory=time.time)
    solvers: list[SolverState] = field(default_factory=list)
    total_cost_usd: float = 0.0
    confirmed_flag: str | None = None
    # FlagStatus value of confirmed_flag. None means a state file written before
    # verification existed: its flag was only a regex match and is not trusted.
    confirmed_status: str | None = None
    winner_id: str | None = None

    def save(self, log_dir: str) -> None:
        out = Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "session-state.json"
        data = {
            "challenge_name": self.challenge_name,
            "challenge_dir": self.challenge_dir,
            "description": self.description,
            "category": self.category,
            "started_at": self.started_at,
            "total_cost_usd": self.total_cost_usd,
            "confirmed_flag": self.confirmed_flag,
            "confirmed_status": self.confirmed_status,
            "winner_id": self.winner_id,
            "solvers": [asdict(s) for s in self.solvers],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, log_dir: str) -> SessionState:
        path = Path(log_dir) / "session-state.json"
        data = json.loads(path.read_text())
        solvers = [SolverState(**s) for s in data.pop("solvers", [])]
        return cls(**data, solvers=solvers)
