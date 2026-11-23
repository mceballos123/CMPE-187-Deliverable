#!/usr/bin/env python3
"""Resolve REVIEW verdicts by hand (oracle and judge disagreed).

  python qa_review.py results/run_XXXX

For each REVIEW execution it shows the prompt, the reference key points, both opinions
and the answer, then asks for P (Pass), A (Partial Pass), F (Fail) or S (skip).
Decisions go to review.jsonl in the run folder. results.jsonl is never edited,
so the raw automated verdicts stay as evidence. qa_report.py applies the decisions.
"""

import json
import sys
import textwrap
from pathlib import Path

CHOICES = {"p": "Pass", "a": "Partial Pass", "f": "Fail"}


def load_reviews(run_dir):
    path = Path(run_dir) / "review.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["app"], r["test_id"], r["repeat"])] = r
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    run = Path(argv[0])
    name = "results_regraded.jsonl" if (run / "results_regraded.jsonl").exists() else "results.jsonl"
    rows = [json.loads(x) for x in (run / name).read_text(encoding="utf-8").splitlines() if x.strip()]
    done = load_reviews(run)
    todo = [r for r in rows if r.get("final_verdict") == "REVIEW" and (r["app"], r["test_id"], r["repeat"]) not in done]
    print("%d REVIEW executions to decide (%d already decided)." % (len(todo), len(done)))
    with open(run / "review.jsonl", "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            o, j = r["oracle"], r.get("judge") or {}
            print("\n" + "=" * 78)
            print("%d/%d  %s  %s  repeat %d" % (i, len(todo), r["app_name"], r["test_id"], r["repeat"]))
            print("PROMPT: " + r["prompt"])
            print("KEY POINTS:\n  - " + "\n  - ".join(r["test"]["reference_key_points"]))
            print("ORACLE: %s %s  %s" % (o["output_class"], o["verdict"],
                                        "; ".join(o["errors"] + o["fabrications"]) or
                                        "missing: " + ", ".join(k["key_point"] for k in o["key_points"] if not k["ok"])))
            print("JUDGE:  %s %s  %s" % (j.get("output_class"), j.get("verdict"), j.get("reason", "")))
            print("ANSWER:\n" + textwrap.indent(textwrap.fill(r["answer"][:2500], 100), "  "))
            while True:
                c = input("Your verdict [p/a/f, s = skip, q = quit]: ").strip().lower()
                if c in CHOICES or c in ("s", "q"):
                    break
            if c == "q":
                break
            if c == "s":
                continue
            note = input("Short reason (optional): ").strip()
            f.write(json.dumps({"app": r["app"], "test_id": r["test_id"], "repeat": r["repeat"],
                                "human_verdict": CHOICES[c], "note": note}) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
