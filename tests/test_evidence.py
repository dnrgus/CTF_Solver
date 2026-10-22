"""Evidence ledger: repeated-failure suppression, novelty, handoff, and crash recovery."""

from __future__ import annotations

import json

from ctf_solver.evidence import KIND_HYPOTHESIS, EvidenceLedger, command_key, exit_code_of

FAIL = "cat: no such file\n[exit 1]"
OK = "hello\n"


def _bash(ledger: EvidenceLedger, command: str, output: str, step: int = 1, agent: str = "a") -> bool:
    return ledger.record_tool(agent=agent, step=step, tool="bash", args={"command": command},
                              output=output, command=command)


def test_exit_code_is_read_from_the_do_bash_marker():
    assert exit_code_of(FAIL) == 1
    assert exit_code_of(OK) == 0
    assert exit_code_of("[exit 124]") == 124


def test_command_key_ignores_whitespace_but_not_content():
    assert command_key("ls   -la") == command_key("ls -la")
    assert command_key("ls -la") != command_key("ls -l")


def test_identical_failed_command_is_suppressed_until_new_evidence():
    ledger = EvidenceLedger()
    _bash(ledger, "cat secret", FAIL)
    assert ledger.suppression("cat secret") is not None  # nothing new since the failure

    _bash(ledger, "ls", OK, step=2)  # new novel output
    assert ledger.suppression("cat secret") is None  # now it may be retried


def test_suppression_message_carries_previous_output_and_says_what_to_do():
    ledger = EvidenceLedger()
    _bash(ledger, "cat secret", FAIL)
    note = ledger.suppression("cat   secret")  # whitespace-insensitive
    assert note is not None
    assert "[suppressed]" in note
    assert "no such file" in note
    assert "Change the input or the approach" in note


def test_successful_command_is_never_suppressed():
    ledger = EvidenceLedger()
    _bash(ledger, "echo hi", OK)
    assert ledger.suppression("echo hi") is None


def test_repeated_identical_output_is_not_novel_and_does_not_advance_epoch():
    ledger = EvidenceLedger()
    assert _bash(ledger, "a", "same") is True
    epoch = ledger.epoch
    assert _bash(ledger, "b", "same") is False
    assert ledger.epoch == epoch


def test_refuted_hypothesis_is_listed_in_the_handoff_with_its_cause():
    ledger = EvidenceLedger()
    ledger.record_hypothesis(agent="a", step=3, hypothesis="XOR key is 0x41", verdict="refuted",
                             failure_cause="decrypts to noise", next_step="try RC4")
    brief = ledger.handoff_brief()
    assert "Refuted hypotheses" in brief
    assert "XOR key is 0x41" in brief
    assert "decrypts to noise" in brief


def test_claims_are_labelled_unverified_in_the_handoff():
    ledger = EvidenceLedger()
    ledger.record_claim(agent="a", step=5, text="the flag is in the header")
    brief = ledger.handoff_brief()
    assert "Unverified claims" in brief
    assert "the flag is in the header" in brief
    assert "Observed" not in brief


def test_observed_section_only_contains_successful_commands():
    ledger = EvidenceLedger()
    _bash(ledger, "file chall", "ELF 64-bit", step=1)
    _bash(ledger, "bad", FAIL, step=2)
    brief = ledger.handoff_brief()
    assert "file chall" in brief
    assert "bad" not in brief


def test_paths_are_kept_for_context_compression():
    ledger = EvidenceLedger()
    _bash(ledger, "strings /challenge/distfiles/chall.bin | head", OK)
    assert "/challenge/distfiles/chall.bin" in ledger.handoff_brief()


def test_hypothesis_verdict_must_be_known():
    ledger = EvidenceLedger()
    try:
        ledger.record_hypothesis(agent="a", step=1, hypothesis="x", verdict="maybe")
    except ValueError as e:
        assert "verdict" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_replay_recovers_state_after_interruption(tmp_path):
    """Every state change is in execution_log.jsonl: a fresh process rebuilds the same ledger."""
    log = tmp_path / "execution_log.jsonl"
    live = EvidenceLedger(log)
    _bash(live, "cat secret", FAIL, step=1)
    live.record_hypothesis(agent="a", step=2, hypothesis="H1", verdict="refuted")
    live.record_flag(agent="a", step=3, flag="FLAG{x}", status="candidate")

    resumed = EvidenceLedger.load(log)

    # The flag event after the failure is new evidence, so both ledgers lift suppression alike.
    assert resumed.epoch == live.epoch
    assert resumed.suppression("cat secret") == live.suppression("cat secret") is None
    assert [r["hypothesis"] for r in resumed.refuted] == ["H1"]
    assert resumed.flag_status["FLAG{x}"]["status"] == "candidate"
    assert len(resumed.records) == len(live.records)


def test_replay_keeps_suppression_when_no_new_evidence_followed(tmp_path):
    log = tmp_path / "execution_log.jsonl"
    live = EvidenceLedger(log)
    _bash(live, "cat secret", FAIL, step=1)

    resumed = EvidenceLedger.load(log)

    assert resumed.suppression("cat secret") is not None


def test_replay_tolerates_a_partial_last_line(tmp_path):
    """A killed process can leave half a JSON line; the rest must still load."""
    log = tmp_path / "execution_log.jsonl"
    live = EvidenceLedger(log)
    live.record_claim(agent="a", step=1, text="kept")
    with log.open("a", encoding="utf-8") as f:
        f.write('{"kind": "tool", "agent": "a", "st')  # truncated

    resumed = EvidenceLedger.load(log)

    assert any("kept" in c for c in resumed.claims)


def test_missing_log_gives_an_empty_ledger(tmp_path):
    ledger = EvidenceLedger.load(tmp_path / "none.jsonl")
    assert ledger.records == []
    assert ledger.handoff_brief() == ""


def test_log_is_one_json_object_per_line_with_stable_kinds(tmp_path):
    log = tmp_path / "execution_log.jsonl"
    ledger = EvidenceLedger(log)
    ledger.record_hypothesis(agent="a", step=1, hypothesis="H", verdict="supported")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == KIND_HYPOTHESIS
