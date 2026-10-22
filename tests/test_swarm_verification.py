"""End-to-end swarm scenarios for verification, scheduling, suppression, and resume.

These drive the real run loop with the scripted MockProvider and a fake sandbox,
so they exercise the same code paths as a live run without Docker or an LLM.
"""

from __future__ import annotations

from conftest import FakeSandbox, MockProvider, resp, tool
from test_swarm_behavioral import _bash_turn, _swarm

import ctf_solver.solver.swarm as swarm_mod
from ctf_solver.artifacts import write_run_artifacts
from ctf_solver.evidence import EvidenceLedger
from ctf_solver.sandbox import ExecResult
from ctf_solver.session_state import SessionState
from ctf_solver.solver.solver_base import ResultStatus
from ctf_solver.verification import FlagStatus

COMPUTED = "FLAG{from_fresh_sandbox}"


class ReproSandbox(FakeSandbox):
    """A fake sandbox that runs a solve.py only in the sense of the test: a script
    that contains COMPUTED_FLAG_MARKER prints the computed flag, exit 0."""

    async def exec(self, command: str, timeout_s: int = 300) -> ExecResult:
        if "solve.py" in command:
            script = str(self.files.get("/challenge/workspace/solve.py", ""))
            if "COMPUTED_FLAG_MARKER" in script:
                return ExecResult(exit_code=0, stdout=f"{COMPUTED}\n", stderr="")
            return ExecResult(exit_code=1, stdout="", stderr="no script")
        return await super().exec(command, timeout_s)


def use_repro_sandbox(monkeypatch) -> None:
    """Call AFTER patch_backends: it replaces the fake sandboxes it installed."""
    monkeypatch.setattr(swarm_mod, "HostSandbox", ReproSandbox)
    monkeypatch.setattr(swarm_mod, "DockerSandbox", ReproSandbox)


# --- verification --------------------------------------------------------------


async def test_regex_only_flag_never_ends_the_run(tmp_path, patch_backends):
    """A regex match with no verifier and no solve.py is a candidate. The agent keeps going."""
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("submit_flag", flag="FLAG{looks_right}")]),
        _bash_turn("still working"),
        _bash_turn("still working 2"),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1)

    result = await swarm.run()

    assert result is None
    assert swarm.confirmed_flag is None
    assert swarm.flag_records["FLAG{looks_right}"].status == FlagStatus.CANDIDATE
    assert len(provider.sessions["claude-0"].sent) >= 3  # not cancelled by the candidate


async def test_verifier_rejection_is_final_and_counts_as_wrong(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{rejected}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, verifier_cmd="false")

    await swarm.run()

    record = swarm.flag_records["FLAG{rejected}"]
    assert record.status == FlagStatus.FAILED
    assert record.source == "verifier"
    assert swarm.confirmed_flag is None
    assert swarm.solvers["claude-0"]._wrong_submit_count == 1
    # A later deliberate resubmission of the same rejected flag is refused without a new check.
    msg, verified = await swarm._handle_flag_candidate(swarm.solvers["claude-0"], "FLAG{rejected}")
    assert verified is False
    assert "do not resubmit" in msg


async def test_verifier_outage_is_unknown_and_does_not_end_the_run(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("submit_flag", flag="FLAG{outage}")]),
        _bash_turn("keep going"),
    ]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, verifier_cmd="exit 7")

    result = await swarm.run()

    assert result is None
    assert swarm.flag_records["FLAG{outage}"].status == FlagStatus.UNKNOWN
    assert swarm.confirmed_flag is None


async def test_verifier_acceptance_ends_the_run_as_solved(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{accepted}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, verifier_cmd="true")

    result = await swarm.run()

    assert result is not None
    assert result.status == ResultStatus.SOLVED
    assert swarm.confirmed_status == FlagStatus.REMOTE_CONFIRMED


async def test_reproduced_flag_ends_the_run_as_reproduced(tmp_path, patch_backends, monkeypatch):
    """The agent writes solve.py, then submits. The fresh sandbox reproduces the flag."""
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py",
                              content="# COMPUTED_FLAG_MARKER\nprint(compute())\n")]),
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
    ]})
    patch_backends(provider)
    use_repro_sandbox(monkeypatch)
    swarm = _swarm(tmp_path, claude_count=1)

    result = await swarm.run()

    assert result is not None
    assert result.status == ResultStatus.REPRODUCED
    assert swarm.confirmed_status == FlagStatus.LOCALLY_REPRODUCED
    assert swarm.solvers["claude-0"].verified_source is not None
    assert "COMPUTED_FLAG_MARKER" in swarm.solvers["claude-0"].verified_source


async def test_hardcoded_solve_script_does_not_reproduce(tmp_path, patch_backends, monkeypatch):
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("write_file", path="/challenge/workspace/solve.py",
                              content=f"print('{COMPUTED}')  # COMPUTED_FLAG_MARKER\n")]),
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
        _bash_turn("done"),
    ]})
    patch_backends(provider)
    use_repro_sandbox(monkeypatch)
    swarm = _swarm(tmp_path, claude_count=1)

    result = await swarm.run()

    assert result is None
    assert swarm.flag_records[COMPUTED].status == FlagStatus.FAILED
    assert "hardcoded" in swarm.flag_records[COMPUTED].notes[-1]


async def test_resubmit_with_unchanged_solve_script_is_refused(tmp_path, patch_backends, monkeypatch):
    """No solve.py yet: the first check is recorded; resubmitting with the same (absent) script is refused."""
    provider = MockProvider({"claude-0": [
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
        resp(tool_calls=[tool("submit_flag", flag=COMPUTED)]),
    ]})
    patch_backends(provider)
    use_repro_sandbox(monkeypatch)
    swarm = _swarm(tmp_path, claude_count=1)

    await swarm.run()

    inst = swarm.solvers["claude-0"]
    assert swarm.flag_records[COMPUTED].checked_digest  # the first check was recorded
    msg, verified = await swarm._handle_flag_candidate(inst, COMPUTED)
    assert verified is False
    assert "has not changed since the last check" in msg


# --- scheduling ----------------------------------------------------------------


async def test_default_single_agent_never_starts_a_second(tmp_path, patch_backends):
    provider = MockProvider({
        "claude-0": [resp(text="nothing here")],
        "codex-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{second}")])],
    })
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, codex_count=1, max_bumps=0, verifier_cmd="true")

    result = await swarm.run()

    assert result is None
    assert "codex-0" not in provider.sessions


async def test_refuted_first_agent_hands_over_to_a_second_agent(tmp_path, patch_backends):
    provider = MockProvider({
        "claude-0": [resp(text="I could not find it")],
        "codex-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{second_agent}")])],
    })
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, codex_count=1, max_agents=2, max_bumps=0, verifier_cmd="true")

    result = await swarm.run()

    assert result is not None
    assert result.status == ResultStatus.SOLVED
    assert result.solver_id == "codex-0"
    reasons = [s["reason"] for s in swarm.evidence.spawns]
    assert reasons == ["initial", "refuted"]


# --- suppression ---------------------------------------------------------------


class FailingCatSandbox(FakeSandbox):
    """Every `cat secret` fails the way a missing file does."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.exec_map = {"cat secret": ExecResult(exit_code=1, stdout="", stderr="no such file")}


async def test_identical_failed_command_is_not_rerun_without_new_evidence(tmp_path, patch_backends, monkeypatch):
    provider = MockProvider({"claude-0": [
        _bash_turn("cat secret"),
        _bash_turn("cat secret"),
        resp(text="done"),
    ]})
    patch_backends(provider)
    monkeypatch.setattr(swarm_mod, "HostSandbox", FailingCatSandbox)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=0)

    await swarm.run()

    # The second identical call never reached the sandbox: it was suppressed, not counted as a run.
    assert swarm.evidence.bash_calls == 1
    assert swarm.evidence.suppressed_calls == 1
    second_results = provider.sessions["claude-0"].sent[2][1]
    assert any("[suppressed]" in str(r.content) for r in second_results)


# --- resume --------------------------------------------------------------------


async def test_resumed_run_receives_the_recorded_refutations(tmp_path, patch_backends):
    first = MockProvider({"claude-0": [
        resp(tool_calls=[tool("record_experiment", hypothesis="key is 0x41", verdict="refuted",
                              failure_cause="decrypts to noise")]),
    ]})
    patch_backends(first)
    swarm1 = _swarm(tmp_path, claude_count=1, max_steps=1, max_bumps=0)
    await swarm1.run()

    second = MockProvider({"claude-0": [resp(text="done")]})
    patch_backends(second)
    swarm2 = _swarm(tmp_path, claude_count=1, max_bumps=0)
    swarm2.resume_state = SessionState(
        challenge_name="test", challenge_dir=str(tmp_path), description="solve me", category="misc",
    )
    await swarm2.run()

    first_message = second.sessions["claude-0"].sent[0][0]
    assert "key is 0x41" in first_message
    assert "decrypts to noise" in first_message
    assert "independent agent" not in first_message  # the first agent resumes, it is not a replacement


async def test_resume_loads_candidate_status_so_a_rejected_flag_stays_rejected(tmp_path, patch_backends):
    first = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{nope}")])]})
    patch_backends(first)
    swarm1 = _swarm(tmp_path, claude_count=1, max_bumps=0, verifier_cmd="false")
    await swarm1.run()

    second = MockProvider({"claude-0": [resp(text="done")]})
    patch_backends(second)
    swarm2 = _swarm(tmp_path, claude_count=1, max_bumps=0, verifier_cmd="false")
    swarm2.resume_state = SessionState(
        challenge_name="test", challenge_dir=str(tmp_path), description="solve me", category="misc",
    )
    await swarm2.run()

    assert swarm2.flag_records["FLAG{nope}"].status == FlagStatus.FAILED
    assert swarm2.flag_records["FLAG{nope}"].finally_rejected


# --- artifacts -----------------------------------------------------------------


async def test_artifacts_for_an_unsolved_run_make_no_solve_py(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(text="no luck")]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, max_bumps=0)
    await swarm.run()

    names = write_run_artifacts(swarm, "test", "misc")

    assert "solve.py" not in names
    assert "writeup.md" in names
    assert (tmp_path / "verification.json").is_file()
    assert (tmp_path / "evidence.json").is_file()
    assert (tmp_path / "execution_log.jsonl").is_file()


def test_evidence_ledger_is_empty_for_a_fresh_swarm(tmp_path):
    swarm = _swarm(tmp_path, claude_count=1)
    assert isinstance(swarm.evidence, EvidenceLedger)
    assert swarm.evidence.records == []


async def test_no_agent_starts_once_the_run_is_cancelled(tmp_path, patch_backends):
    """A Ctrl-C must not start a new agent, even if the remaining agent just finished."""
    provider = MockProvider({"claude-0": [resp(text="gave up")], "codex-0": []})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, codex_count=1, max_agents=2, max_bumps=0)
    swarm._plan = swarm._create_solvers()[1:]  # codex-0 is planned only
    swarm.cancel_event.set()
    tasks: dict = {}

    swarm._maybe_spawn(tasks)

    assert tasks == {}
    assert "codex-0" not in provider.sessions


async def test_hard_kill_after_verification_resumes_as_solved(tmp_path, patch_backends):
    """A verified flag reached the ledger, but the process died before saving session state.

    The resumed run must trust the ledger: it reports the flag and starts no agent."""
    first = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{before_crash}")])]})
    patch_backends(first)
    swarm1 = _swarm(tmp_path, claude_count=1, verifier_cmd="true")
    await swarm1.run()
    assert swarm1.confirmed_flag == "FLAG{before_crash}"
    (tmp_path / "session-state.json").unlink(missing_ok=True)  # simulate the crash

    second = MockProvider({"claude-0": [resp(text="should not run")]})
    patch_backends(second)
    swarm2 = _swarm(tmp_path, claude_count=1, verifier_cmd="true")
    swarm2.resume_state = SessionState(
        challenge_name="test", challenge_dir=str(tmp_path), description="solve me", category="misc",
    )
    result = await swarm2.run()

    assert result is None
    assert second.sessions == {}
    assert swarm2.confirmed_flag == "FLAG{before_crash}"
    assert swarm2.confirmed_status == FlagStatus.REMOTE_CONFIRMED


async def test_record_names_who_decided_the_verdict(tmp_path, patch_backends):
    provider = MockProvider({"claude-0": [resp(tool_calls=[tool("submit_flag", flag="FLAG{who}")])]})
    patch_backends(provider)
    swarm = _swarm(tmp_path, claude_count=1, verifier_cmd="true")

    await swarm.run()

    assert swarm.flag_records["FLAG{who}"].source == "verifier"
    assert swarm.evidence.flag_status["FLAG{who}"]["source"] == "verifier"
