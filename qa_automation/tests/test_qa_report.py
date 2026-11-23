"""Checks on the statistics: repeatability, verdict counts and the final-verdict rule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qa_judge  # noqa: E402
import qa_report  # noqa: E402

NAMES = {"chatgpt": "ChatGPT", "claude": "Claude", None: "All"}


def row(app, test, rep, verdict):
    return {"app": app, "test_id": test, "repeat": rep, "final_verdict": verdict}


def test_repeatability_finds_changed_verdicts():
    rows = [row("chatgpt", "QA-01-ORIG-CX01", 1, "Pass"), row("chatgpt", "QA-01-ORIG-CX01", 2, "Pass"),
            row("chatgpt", "QA-06-ORIG-CX01", 1, "Pass"), row("chatgpt", "QA-06-ORIG-CX01", 2, "Partial Pass"),
            row("claude", "QA-01-ORIG-CX01", 1, "Fail")]
    table, flaky = qa_report.repeatability_tables(rows, ["chatgpt", "claude"], NAMES)
    assert table[0] == {"app": "ChatGPT", "tests_with_2plus_repeats": 2, "same_verdict_every_repeat": 1,
                        "stability": "50.0%"}
    assert table[1]["tests_with_2plus_repeats"] == 0
    assert flaky == [{"app": "ChatGPT", "test": "QA-06-ORIG-CX01", "verdicts": "Pass / Partial Pass"}]


def test_verdict_table_counts_and_rates():
    rows = [row("chatgpt", "a", 1, "Pass"), row("chatgpt", "b", 1, "Fail"), row("chatgpt", "c", 1, "NOT RUN"),
            row("chatgpt", "d", 1, "Partial Pass")]
    t = qa_report.verdict_table(rows, lambda r: r["app"], ["chatgpt"])
    assert t[0]["executed"] == 3 and t[0]["pass"] == 1 and t[0]["not_run"] == 1
    assert t[0]["pass_rate"] == "33.3%" and t[0]["pass_or_partial"] == "66.7%"
    assert t[-1]["group"] == "Total"


def test_final_verdict_rule():
    oracle = {"verdict": "Pass"}
    assert qa_judge.combine(oracle, None) == ("Pass", "oracle-only")
    assert qa_judge.combine(oracle, {"status": "unavailable"}) == ("Pass", "oracle-only")
    assert qa_judge.combine(oracle, {"status": "ok", "verdict": "Pass"}) == ("Pass", "agree")
    assert qa_judge.combine(oracle, {"status": "ok", "verdict": "Partial Pass"}) == ("REVIEW", "disagree")
