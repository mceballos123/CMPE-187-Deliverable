#!/usr/bin/env python3
"""Second-opinion grading with a local LLM (Ollama), plus re-grading of saved runs.

The rule-based oracle (qa_oracle.py) is the main validator. The judge reads the same
reference key points and picks an output class too. If both give the same verdict it
stands. If they disagree the execution is marked REVIEW and a person decides with
qa_review.py. If Ollama is not running, the result says so (judge status "unavailable")
and only the oracle verdict is used.

  python qa_judge.py check                         # is Ollama up and is the model pulled?
  python qa_judge.py regrade results/run_XXXX      # re-run the oracle (and judge) on saved answers
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import qa_oracle
import qa_sites

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_URL = "http://localhost:11434"

CLASSES_TEXT = "\n".join("%s %s (%s)" % (k, v["name"], v["verdict"]) for k, v in {
    "QA-OC-01": {"name": "Accurate, relevant and complete", "verdict": "Pass"},
    "QA-OC-02": {"name": "Says the subject does not exist, invents nothing, gives the correct related info",
                 "verdict": "Pass"},
    "QA-OC-03": {"name": "Accurate but a key point is missing", "verdict": "Partial Pass"},
    "QA-OC-04": {"name": "Accurate but buried in a lot of unrelated filler", "verdict": "Partial Pass"},
    "QA-OC-05": {"name": "Contains a wrong fact, value, command option or calculation", "verdict": "Fail"},
    "QA-OC-06": {"name": "Invents a standard, RFC, value, feature or option that does not exist", "verdict": "Fail"},
    "QA-OC-07": {"name": "Answers a different question or misses most key points", "verdict": "Fail"},
}.items())

PROMPT = """You grade chatbot answers for a software testing project. Be strict and literal.

QUESTION:
{prompt}

REFERENCE KEY POINTS (a correct answer covers all of them):
{kps}

PASS CRITERIA: {criteria}

OUTPUT CLASSES:
{classes}

A fully correct answer to this question belongs in {expected}.

CHATBOT ANSWER:
<<<
{answer}
>>>

Pick exactly one output class for the chatbot answer. Reply with JSON only, like
{{"output_class": "QA-OC-01", "reason": "one short sentence"}}"""


def judge(answer, test, model=DEFAULT_MODEL, url=DEFAULT_URL, timeout=120):
    """Ask Ollama for an output class. Never raises: returns a dict with a status."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            prompt=test["prompt"], kps="\n".join("- " + k for k in test["reference_key_points"]),
            criteria=test.get("pass_criteria", ""), classes=CLASSES_TEXT,
            expected=test["expected_output"], answer=(answer or "")[:6000])}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
    }
    t0 = time.time()
    try:
        req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.load(r)["message"]["content"]
    except Exception as e:  # Ollama not running, model missing, timeout
        return {"status": "unavailable", "model": model, "error": str(e)[:200], "secs": round(time.time() - t0, 2)}
    try:
        parsed = json.loads(content)
        oc = str(parsed.get("output_class", "")).strip().upper()
        if oc not in qa_oracle.OC_VERDICT:
            raise ValueError("bad class %r" % oc)
    except Exception as e:
        return {"status": "invalid", "model": model, "raw": content[:300], "error": str(e)[:200],
                "secs": round(time.time() - t0, 2)}
    return {"status": "ok", "model": model, "output_class": oc, "verdict": qa_oracle.OC_VERDICT[oc],
            "reason": str(parsed.get("reason", ""))[:300], "secs": round(time.time() - t0, 2)}


def combine(oracle_result, judge_result):
    """Final verdict. Returns (verdict, agreement) where agreement is agree / disagree / oracle-only."""
    v = oracle_result["verdict"]
    if not judge_result or judge_result.get("status") != "ok":
        return v, "oracle-only"
    if judge_result["verdict"] == v:
        return v, "agree"
    return "REVIEW", "disagree"


def grade(answer, test, use_judge=True, model=DEFAULT_MODEL, url=DEFAULT_URL):
    o = qa_oracle.evaluate(answer, test)
    j = judge(answer, test, model, url) if use_judge else None
    final, agreement = combine(o, j)
    return o, j, final, agreement


def cmd_check(args):
    test = {"prompt": "Which SNMP operation retrieves a large table in fewer requests, and in which version was it "
                      "introduced?", "reference_key_points": ["GetBulkRequest", "Introduced in SNMPv2"],
            "pass_criteria": "Both points correct.", "expected_output": "QA-OC-01"}
    r = judge("GetBulkRequest, introduced in SNMPv2.", test, args.model, args.url)
    print(json.dumps(r, indent=2))
    return 0 if r["status"] == "ok" else 1


def cmd_regrade(args):
    run = Path(args.run_dir)
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    out, changed = [], 0
    for r in rows:
        if r.get("status") == "ok" and qa_sites.is_site_error(r.get("answer")):
            r.update(status="error", error="the site returned an error message instead of an answer",
                     oracle=None, judge=None, final_verdict="NOT RUN", agreement=None, regraded=True)
        elif r.get("status") == "ok":
            o, j, final, agreement = grade(r["answer"], r["test"], args.judge, args.model, args.url)
            if j is None and r.get("judge") and not args.judge:
                j = r["judge"]  # keep the saved judge opinion when only the oracle is re-run
                final, agreement = combine(o, j)
            if final != r.get("final_verdict"):
                changed += 1
            r.update(oracle=o, judge=j, final_verdict=final, agreement=agreement, regraded=True)
        out.append(r)
    dest = run / "results_regraded.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("Re-graded %d executions, %d final verdicts changed. Written to %s" % (len(out), changed, dest))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=DEFAULT_URL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    rg = sub.add_parser("regrade")
    rg.add_argument("run_dir")
    rg.add_argument("--judge", action="store_true", help="also ask the judge again (default: oracle only)")
    args = ap.parse_args()
    return {"check": cmd_check, "regrade": cmd_regrade}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
