"""Tests for SessionState / SolverState save↔load round-trips."""

import pytest

from ctf_solver.session_state import SessionState, SolverState


def _sample_session(log_dir: str) -> SessionState:
    return SessionState(
        challenge_name="pwn-overflow",
        challenge_dir=log_dir,
        description="A classic stack smash.",
        category="pwn",
        started_at=1700000000.0,
        total_cost_usd=1.2345,
        confirmed_flag="flag{abc}",
        winner_id="solver-2",
        solvers=[
            SolverState(
                solver_id="solver-1",
                provider="claude-sdk",
                status="failed",
                steps=7,
                cost_usd=0.5,
                flag=None,
                error="ran out of ideas",
            ),
            SolverState(
                solver_id="solver-2",
                provider="codex",
                status="solved",
                steps=12,
                cost_usd=0.7345,
                flag="flag{abc}",
                error=None,
            ),
            SolverState(
                solver_id="solver-3",
                provider="zai",
                status="timeout",
                steps=20,
                cost_usd=0.0,
            ),
        ],
    )


def test_save_load_round_trip_preserves_all_fields(tmp_path):
    original = _sample_session(str(tmp_path))
    original.save(str(tmp_path))

    loaded = SessionState.load(str(tmp_path))

    # Challenge-level fields.
    assert loaded.challenge_name == "pwn-overflow"
    assert loaded.challenge_dir == str(tmp_path)
    assert loaded.description == "A classic stack smash."
    assert loaded.category == "pwn"
    assert loaded.started_at == 1700000000.0
    assert loaded.total_cost_usd == 1.2345
    assert loaded.confirmed_flag == "flag{abc}"
    assert loaded.winner_id == "solver-2"

    # Solvers list with real status strings preserved in order.
    assert [s.status for s in loaded.solvers] == ["failed", "solved", "timeout"]
    assert [s.solver_id for s in loaded.solvers] == ["solver-1", "solver-2", "solver-3"]
    assert loaded.solvers[1].provider == "codex"
    assert loaded.solvers[1].flag == "flag{abc}"
    assert loaded.solvers[1].steps == 12
    assert loaded.solvers[1].cost_usd == 0.7345
    assert loaded.solvers[0].error == "ran out of ideas"
    assert all(isinstance(s, SolverState) for s in loaded.solvers)


def test_solved_solver_survives_round_trip(tmp_path):
    """resume's skip logic reads status=='solved'; ensure it round-trips intact."""
    session = SessionState(
        challenge_name="c",
        challenge_dir=str(tmp_path),
        description="d",
        category="rev",
        solvers=[
            SolverState(solver_id="winner", provider="codex", status="solved", flag="flag{x}"),
        ],
    )
    session.save(str(tmp_path))

    loaded = SessionState.load(str(tmp_path))
    assert len(loaded.solvers) == 1
    assert loaded.solvers[0].status == "solved"
    assert loaded.solvers[0].flag == "flag{x}"


def test_load_missing_file_raises(tmp_path):
    missing_dir = tmp_path / "no-session-here"
    missing_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        SessionState.load(str(missing_dir))
