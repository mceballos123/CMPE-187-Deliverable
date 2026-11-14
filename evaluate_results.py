#!/usr/bin/env python3
"""
Evaluate saved results: can each LLM retain information from the initial prompt?

Reads results.json files written by llm_memory_test.py and prints, per LLM:
    Claude --> passed 7/7 --> 100%

Two checks per run:
  1. Keyword check  - the pass/fail already stored by llm_memory_test.py
  2. Local LLM judge - Ollama (free, offline) reads the whole conversation and decides
     whether the final answer correctly uses information from earlier in the chat.
Verdict: PASS if both agree on PASS, FAIL if both agree on FAIL, REVIEW if they disagree
(REVIEW runs are NOT counted as passed - check them by hand).

Usage:
    python evaluate_results.py                     # merge all runs, latest result per test
    python evaluate_results.py results/run_X/results.json [more.json ...]
    python evaluate_results.py --no-judge          # keyword check only (no Ollama needed)
    python evaluate_results.py --model llama3.2:3b
Output: results/evaluation.json, results/evaluation.csv
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/chat"

JUDGE_PROMPT = """You are an AI test judge. Review this 5-turn conversation where the model ingested facts in Turn 1, processed 3 intervening prompts, and was tested on recall in Turn 5.

--- CONVERSATION ---
{conversation_log}
--------------------

EXPECTED CONTEXT (Turn 1): {expected_context}
TARGET RECALL QUERY (Turn 5): {expected_oracle}

Evaluate:
1. Did the model retain the core information from Turn 1 despite the 3 intervening prompts?
2. Does the final response accurately answer the recall query without hallucinating?

Output ONLY valid JSON:
{
  "reasoning": "Short explanation of memory retention across turns.",
  "verdict": "PASS" // or "FAIL"
}"""


def build_conversation_log(rec):
    """Number every turn: User/Model pairs from the saved context, then the final follow-up + response."""
    turns = json.loads(rec.get("conversation_context") or "[]")
    turns += [{"role": "user", "text": rec["follow_up_prompt"]},
              {"role": "assistant", "text": rec["actual_response"][:6000]}]
    lines, n = [], 0
    for t in turns:
        if t["role"] == "user":
            n += 1
        lines.append(f"[Turn {n} - {'User' if t['role'] == 'user' else 'Model'}]: {t['text']}")
    return "\n\n".join(lines)


def load_results(paths):
    """Merge result files; keep the latest run per (llm, test_id, filler_count)."""
    latest = {}
    for p in paths:
        for r in json.loads(Path(p).read_text(encoding="utf-8")):
            key = (r["llm"], r["test_id"], r.get("filler_count", 0))
            if key not in latest or r["timestamp"] >= latest[key]["timestamp"]:
                latest[key] = r
    return sorted(latest.values(), key=lambda r: (r["llm"], r["test_id"], r.get("filler_count", 0)))


def judge(rec, model):
    """Ask Ollama; returns (verdict 'PASS'/'FAIL'/'ERROR', reason)."""
    context_prompts = rec["initial_prompt"] + (f" / {rec['second_prompt']}" if rec.get("second_prompt") else "")
    prompt = (JUDGE_PROMPT  # str.replace, not .format: the template contains literal JSON braces
              .replace("{conversation_log}", build_conversation_log(rec))
              .replace("{expected_context}", context_prompts)
              .replace("{expected_oracle}", f"{rec['follow_up_prompt']} | Expected: {rec['expected_result']} "
                                            f"| Pass criteria: {rec['pass_criteria']}"))
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(json.loads(resp.read())["message"]["content"])
        verdict = str(out.get("verdict", "")).upper()
        return (verdict if verdict in ("PASS", "FAIL") else "ERROR"), str(out.get("reasoning", out.get("reason", "")))[:300]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        return "ERROR", f"{type(e).__name__}: {e}"


def final_verdict(keyword, judged):
    if keyword not in ("PASS", "FAIL"):  # the test run itself errored (no response to grade)
        return "ERROR"
    if judged is None or judged == "ERROR":  # judge off/unavailable -> keyword check decides
        return keyword
    return keyword if keyword == judged else "REVIEW"


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="results.json files (default: all runs under results/)")
    ap.add_argument("--no-judge", action="store_true", help="Skip the Ollama judge")
    ap.add_argument("--fillers", default="3", choices=["0", "3", "all"],
                    help="Which condition to report: 3 = with 3 unrelated prompts in between (default), "
                         "0 = no fillers, all = everything")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model for the judge")
    args = ap.parse_args()

    files = args.files or sorted(str(p) for p in (BASE_DIR / "results").glob("run_*/results.json"))
    if not files:
        sys.exit("No results.json files found.")
    records = load_results(files)
    if args.fillers != "all":
        records = [r for r in records if str(r.get("filler_count", 0)) == args.fillers]

    rows = []
    for i, r in enumerate(records, 1):
        if sys.stderr.isatty():
            print(f"\r  grading {i}/{len(records)}: {r['llm']} {r['test_id']}        ", end="", file=sys.stderr, flush=True)
        errored = r["result"] not in ("PASS", "FAIL")  # run produced no usable response
        j, why = (None, "") if (args.no_judge or errored) else judge(r, args.model)
        rows.append({
            "llm": r["llm"], "test_id": r["test_id"], "filler_count": r.get("filler_count", 0),
            "collection_method": r.get("collection_method", "automated"),
            "keyword_result": r["result"], "judge_result": j or "SKIPPED", "judge_reason": why,
            "final_verdict": final_verdict(r["result"], j), "keyword_explanation": r["failure_explanation"],
            "timestamp": r["timestamp"],
        })
    if sys.stderr.isatty():
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)

    out_json = BASE_DIR / "results" / "evaluation.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(BASE_DIR / "results" / "evaluation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print_report(rows, args)
    print(f"  Saved: {out_json.relative_to(BASE_DIR)} and evaluation.csv\n")


# ---------------------------------------------------------------- pretty report
COLOR = sys.stdout.isatty()
CODES = {"green": "32", "red": "31", "yellow": "33", "grey": "90", "bold": "1", "cyan": "36"}
LLM_ORDER = ["ChatGPT", "Claude", "Gemini", "Perplexity"]


def c(text, color):
    return f"\033[{CODES[color]}m{text}\033[0m" if COLOR else text


def cell(verdict, manual):
    label = {"PASS": "PASS", "FAIL": "FAIL", "REVIEW": "REVIEW", "ERROR": "ERROR"}.get(verdict, "-")
    plain = label + ("*" if manual else "")
    color = {"PASS": "green", "FAIL": "red", "REVIEW": "yellow"}.get(verdict, "grey")
    return plain, color


def bar(passed, total, width=20):
    filled = round(width * passed / total) if total else 0
    return c("█" * filled, "green") + c("░" * (width - filled), "grey")


def print_report(rows, args):
    llms = [l for l in LLM_ORDER if any(r["llm"] == l for r in rows)] + \
           sorted({r["llm"] for r in rows} - set(LLM_ORDER))
    tests = sorted({r["test_id"] for r in rows})
    look = {(r["llm"], r["test_id"]): r for r in rows}
    cond = "all runs" if args.fillers == "all" else (
        "no filler prompts" if args.fillers == "0" else f"{args.fillers} unrelated prompts between the setup and the recall question")
    W = 66
    print()
    print(c("╔" + "═" * W + "╗", "cyan"))
    print(c("║", "cyan") + c("  LLM MEMORY RETENTION REPORT".ljust(W), "bold") + c("║", "cyan"))
    print(c("║", "cyan") + f"  Condition: {cond}"[:W].ljust(W) + c("║", "cyan"))
    judge_txt = "keyword check only" if args.no_judge else f"keyword check + local judge ({args.model})"
    print(c("║", "cyan") + f"  Grading:   {judge_txt}"[:W].ljust(W) + c("║", "cyan"))
    print(c("╚" + "═" * W + "╝", "cyan"))

    # ---- results by test
    w0, wc = 8, 12
    line = lambda l, m_, r_: l + "─" * (w0 + 2) + "".join(m_ + "─" * (wc + 2) for _ in llms) + r_
    print("\n" + c("  RESULTS BY TEST", "bold"))
    print("  " + line("┌", "┬", "┐"))
    print("  │ " + "Test".ljust(w0) + " │" + "".join(" " + l.center(wc) + " │" for l in llms))
    print("  " + line("├", "┼", "┤"))
    for t in tests:
        out = f"  │ {t.ljust(w0)} │"
        for l in llms:
            r = look.get((l, t))
            plain, color = cell(r["final_verdict"], r["collection_method"] == "manual") if r else ("-", "grey")
            out += " " + c(plain.center(wc), color) + " │"
        print(out)
    print("  " + line("└", "┴", "┘"))

    # ---- score by LLM
    print("\n" + c("  MEMORY RETENTION SCORE", "bold"))
    notes = []
    for l in llms:
        rs = [r for r in rows if r["llm"] == l and r["final_verdict"] != "ERROR"]
        passed = sum(r["final_verdict"] == "PASS" for r in rs)
        review = sum(r["final_verdict"] == "REVIEW" for r in rs)
        pc = 100 * passed / len(rs) if rs else 0
        color = "green" if pc >= 80 else ("yellow" if pc >= 50 else "red")
        print(f"  {l.ljust(11)} {bar(passed, len(rs))}  {c(f'{passed}/{len(rs)}'.rjust(4), color)}  "
              f"{c(f'{pc:.0f}%'.rjust(4), color)}")
        if review:
            notes.append(f"{l}: {review} run(s) need manual review")
        missing = [t for t in tests if (l, t) not in look]
        if missing:
            notes.append(f"{l}: no result yet for {', '.join(missing)}")
        errors = len([r for r in rows if r["llm"] == l]) - len(rs)
        if errors:
            notes.append(f"{l}: {errors} run(s) errored and were not scored")

    # ---- legend
    print("\n" + c("  PASS", "green") + " = recalled the earlier information   " + c("FAIL", "red") + " = did not")
    print(c("  REVIEW", "yellow") + " = keyword check and judge disagree (not counted as a pass)")
    if any(r["collection_method"] == "manual" for r in rows):
        print("  *      = response copied by hand from the website")
    for n in notes:
        print(c(f"  ! {n}", "yellow"))
    print()


if __name__ == "__main__":
    main()
