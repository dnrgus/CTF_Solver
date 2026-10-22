"""Independent reproduction of solve.py in a fresh sandbox."""

from __future__ import annotations

from conftest import FakeSandbox

from ctf_solver.reproduce import SOLVE_SCRIPT_PATH, reproduce_candidate
from ctf_solver.sandbox import ExecResult
from ctf_solver.verification import FlagStatus

FLAG = "FLAG{r3pr0}"


class _Factory:
    """Hands out fresh FakeSandboxes and remembers each one for inspection."""

    def __init__(self, exec_result: ExecResult | None = None, fail_start: bool = False) -> None:
        self.created: list[FakeSandbox] = []
        self.exec_result = exec_result
        self.fail_start = fail_start

    async def __call__(self) -> FakeSandbox:
        sandbox = FakeSandbox()
        if self.exec_result is not None:
            sandbox.default_exec = self.exec_result
        if self.fail_start:
            async def boom() -> None:
                raise RuntimeError("docker is down")
            sandbox.start = boom
        self.created.append(sandbox)
        return sandbox


async def test_no_solve_script_stays_candidate():
    status, note = await reproduce_candidate(FLAG, None, _Factory())
    assert status == FlagStatus.CANDIDATE
    assert "no solve.py" in note


async def test_script_that_prints_flag_in_fresh_sandbox_is_reproduced():
    factory = _Factory(ExecResult(exit_code=0, stdout=f"{FLAG}\n", stderr=""))
    status, note = await reproduce_candidate(FLAG, "print('computed')", factory)
    assert status == FlagStatus.LOCALLY_REPRODUCED
    assert len(factory.created) == 1
    assert factory.created[0].files[SOLVE_SCRIPT_PATH] == "print('computed')"
    assert factory.created[0].stopped


async def test_hardcoded_flag_literal_is_not_a_reproduction():
    factory = _Factory(ExecResult(exit_code=0, stdout=f"{FLAG}\n", stderr=""))
    status, note = await reproduce_candidate(FLAG, f"print('{FLAG}')", factory)
    assert status == FlagStatus.FAILED
    assert "hardcoded" in note
    assert factory.created == []  # never ran


async def test_script_that_exits_nonzero_is_failed():
    factory = _Factory(ExecResult(exit_code=1, stdout=f"{FLAG}", stderr="Traceback"))
    status, note = await reproduce_candidate(FLAG, "raise SystemExit(1)", factory)
    assert status == FlagStatus.FAILED
    assert "exit 1" in note


async def test_script_that_prints_other_text_is_failed():
    factory = _Factory(ExecResult(exit_code=0, stdout="nothing useful", stderr=""))
    status, _ = await reproduce_candidate(FLAG, "print('x')", factory)
    assert status == FlagStatus.FAILED


async def test_fresh_sandbox_start_failure_is_unknown_not_rejected():
    factory = _Factory(fail_start=True)
    status, note = await reproduce_candidate(FLAG, "print('x')", factory)
    assert status == FlagStatus.UNKNOWN
    assert "docker is down" in note


async def test_failure_on_a_remote_challenge_says_reachability_was_not_checked():
    factory = _Factory(ExecResult(exit_code=1, stdout="", stderr="connection refused"))
    status, note = await reproduce_candidate(FLAG, "print('x')", factory, remote_target="nc 10.0.0.5 1337")
    assert status == FlagStatus.FAILED
    assert "reachability" in note


async def test_failure_on_a_local_challenge_has_no_remote_caveat():
    factory = _Factory(ExecResult(exit_code=1, stdout="", stderr="boom"))
    _, note = await reproduce_candidate(FLAG, "print('x')", factory)
    assert "reachability" not in note
