#!/usr/bin/env python3
"""
Grading + results table for the NLC tests. nlc_test.py calls this automatically at the end
of a test run; nothing is saved to disk, the results are printed in the terminal.

Two independent checks per answer:
  1. Keyword check   - the word/phrase rules in nlc_test.py
  2. Local AI judge  - a small AI model running on this computer through Ollama (free, offline)
                       reads the question, the expected result and the output-class definitions,
                       and decides which output class (NL-OC-xx) the answer belongs to.
Final verdict:  both agree -> that verdict;  they disagree -> REVIEW (a person decides);
                the app gave no answer -> BLOCKED (rerun it);  --no-judge -> keyword decides.
NLC-02 also gets a consistency check: are the answers to 02a, 02b and 02c the same at their core?
"""
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

from nlc_test import DATA_FILE, nlc02_facts      # (also makes Martin's files importable)
from evaluate_results import c, bar, LLM_ORDER   # Martin's console colours / score bars

OLLAMA_URL = "http://localhost:11434/api/chat"
DATA = json.loads(DATA_FILE.read_text(encoding="utf-8"))
OC = DATA["output_classes"]
IC = DATA["input_classes"]
TEST_CASES = ["NLC-01", "NLC-02", "NLC-03", "NLC-04", "NLC-05"]
COMPLEXITY = {p["test_case"]: p["complexity"] for p in DATA["prompts"] if not p["augmented"]}


# --------------------------------------------------------------------------- #
# Talking to the local AI judge (Ollama)
# --------------------------------------------------------------------------- #
def ollama_running():
    try:
        urllib.request.urlopen(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=5)
        return True
    except OSError:
        return False


def ask_ollama(prompt, model, temperature=0.0):
    """Send one prompt to Ollama and return its JSON answer as a Python dict."""
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": temperature, "num_ctx": 8192},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(json.loads(resp.read())["message"]["content"])


JUDGE_PROMPT = """You are grading a software test of how well a chatbot understands natural language.
The topic is computer network management. The user's message was sent in a brand-new chat.

USER'S MESSAGE:
{prompt}

TYPE OF MESSAGE: {input_class}
WHAT A GOOD ANSWER DOES: {expected}
PASS CRITERIA: {criteria}

OUTPUT CLASSES YOU MAY CHOOSE FROM:
{classes}

CHATBOT'S ANSWER:
<<<
{response}
>>>

Pick the ONE output class that best describes the chatbot's answer. If you pick a PASS class the
verdict is PASS; if you pick a FAIL class the verdict is FAIL. Judge the meaning, not the length or style.

Output ONLY valid JSON:
{"output_class": "NL-OC-xx", "verdict": "PASS or FAIL", "reasoning": "one or two sentences"}"""

CONSISTENCY_PROMPT = """You are grading a software test. The same question was asked to a chatbot in three
different wordings, each in its own brand-new chat. A good chatbot gives the same core answer to all three.

The correct core answer: SNMPv2c relies on community strings that are sent in plain text; SNMPv3 adds
user authentication and encryption.

{answers}

Do all three answers give the same core differences? Differences in length, tone, or extra details are fine.
They are INCONSISTENT if any answer leaves out a core difference, contradicts another answer, or is incorrect.

Output ONLY valid JSON:
{"consistent": true or false, "reasoning": "one or two sentences"}"""


def per_prompt_classes(rec):
    """Output classes that apply to ONE answer (NL-OC-02/07 are about comparing answers, so left out)."""
    exp = [x for x in rec["expected_output_classes"] if x != "NL-OC-02"]
    bad = [x for x in rec["failure_output_classes"] if x != "NL-OC-07"]
    return exp, bad


def judge_answer(rec, model):
    """Returns (verdict, output_class, reasoning). verdict is PASS, FAIL or ERROR."""
    exp, bad = per_prompt_classes(rec)
    classes = "\n".join([f"- {x} {OC[x]['name']} (PASS): {OC[x]['meaning']}" for x in exp] +
                        [f"- {x} {OC[x]['name']} (FAIL): {OC[x]['meaning']}" for x in bad])
    prompt = (JUDGE_PROMPT  # str.replace, not .format, because the template contains JSON braces
              .replace("{prompt}", rec["prompt"])
              .replace("{input_class}", f"{rec['input_class']} {IC[rec['input_class']]}")
              .replace("{expected}", rec["expected_result"])
              .replace("{criteria}", rec["pass_criteria"])
              .replace("{classes}", classes)
              .replace("{response}", rec["actual_response"][:6000]))
    try:
        out = ask_ollama(prompt, model)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        return "ERROR", "", f"Judge unavailable: {type(e).__name__}: {e}"
    oc = str(out.get("output_class", "")).strip().upper()[:8]
    verdict = str(out.get("verdict", "")).strip().upper()
    if oc in exp:          # the chosen class decides the verdict (more reliable than the verdict field)
        verdict = "PASS"
    elif oc in bad:
        verdict = "FAIL"
    else:
        oc = ""
    if verdict not in ("PASS", "FAIL"):
        return "ERROR", "", f"Judge gave an unusable answer: {json.dumps(out)[:200]}"
    return verdict, oc, str(out.get("reasoning", ""))[:400]


def judge_consistency(recs, model):
    """recs = the 02a, 02b, 02c records of one app and one run. Returns (verdict, reasoning)."""
    answers = "\n\n".join(f"WORDING {chr(65 + i)}: {r['prompt']}\nANSWER {chr(65 + i)}:\n<<<\n"
                          f"{r['actual_response'][:3500]}\n>>>" for i, r in enumerate(recs))
    try:
        out = ask_ollama(CONSISTENCY_PROMPT.replace("{answers}", answers), model)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        return "ERROR", f"Judge unavailable: {type(e).__name__}: {e}"
    consistent = out.get("consistent")
    if isinstance(consistent, str):
        consistent = consistent.strip().lower() == "true"
    if not isinstance(consistent, bool):
        return "ERROR", f"Judge gave an unusable answer: {json.dumps(out)[:200]}"
    return ("PASS" if consistent else "FAIL"), str(out.get("reasoning", ""))[:400]


# --------------------------------------------------------------------------- #
# Combining the two checks
# --------------------------------------------------------------------------- #
def combine(keyword, judged):
    """Martin's rule: agree -> that verdict, disagree -> REVIEW, no answer -> BLOCKED."""
    if keyword not in ("PASS", "FAIL"):
        return "BLOCKED"
    if judged in (None, "ERROR"):
        return keyword
    return keyword if keyword == judged else "REVIEW"


def rollup(verdicts):
    """Combine several verdicts into one: any FAIL fails; otherwise BLOCKED, then REVIEW; else PASS."""
    for v in ("FAIL", "BLOCKED", "REVIEW"):
        if v in verdicts:
            return v
    return "PASS"


def grade_prompts(records, model, use_judge):
    """Add the judge's opinion and the final verdict to every answer record."""
    for i, r in enumerate(records, 1):
        progress(f"judging answer {i}/{len(records)}: {r['llm']} {r['id']} run {r['run_index']}")
        j, j_oc, j_why, secs = None, "", "", 0
        if use_judge and r["keyword_result"] != "ERROR":
            start = time.time()
            j, j_oc, j_why = judge_answer(r, model)
            secs = round(time.time() - start, 1)
        final = combine(r["keyword_result"], j)
        r.update({"judge_result": j or "SKIPPED", "judge_output_class": j_oc, "judge_reasoning": j_why,
                  "judge_seconds": secs, "final_verdict": final,
                  "final_output_class": r["keyword_output_class"] if final in ("PASS", "FAIL") else ""})


def grade_consistency(prompt_rows, model, use_judge):
    """NLC-02 consistency check, once per app per run, after all three answers are collected."""
    groups = defaultdict(list)
    for r in prompt_rows:
        if r["consistency_group"] and not r["augmented"]:
            groups[(r["llm"], r["run_index"])].append(r)
    rows = []
    for (llm, run), recs in sorted(groups.items()):
        recs = sorted(recs, key=lambda r: r["id"])
        row = {"llm": llm, "id": "NLC-02 consistency", "run_index": run, "judge_seconds": 0}
        if len(recs) < 3 or any(r["final_verdict"] == "BLOCKED" for r in recs):
            row.update({"keyword_result": "ERROR", "keyword_explanation": "Not all three answers were collected.",
                        "judge_result": "SKIPPED", "judge_reasoning": ""})
        else:
            facts = {r["id"]: nlc02_facts(r["actual_response"]) for r in recs}
            same = len({tuple(f) for f in facts.values()}) == 1
            row["keyword_result"] = "PASS" if same else "FAIL"
            row["keyword_explanation"] = "" if same else "Core facts differ: " + "; ".join(
                f"{k}: {', '.join(v) or 'none'}" for k, v in facts.items())
            progress(f"judging NLC-02 consistency: {llm} run {run}")
            start = time.time()
            j, why = judge_consistency(recs, model) if use_judge else (None, "")
            row.update({"judge_result": j or "SKIPPED", "judge_reasoning": why,
                        "judge_seconds": round(time.time() - start, 1) if j else 0})
        final = combine(row["keyword_result"], None if row["judge_result"] == "SKIPPED" else row["judge_result"])
        row["final_verdict"] = final
        row["final_output_class"] = {"PASS": "NL-OC-02", "FAIL": "NL-OC-07"}.get(final, "")
        rows.append(row)
    return rows


def effective_verdict(row, cons):
    """A single NLC-02 prompt only meets its rule (R2-R4) if it passes AND the three wordings are consistent."""
    if row["consistency_group"] and not row["augmented"]:
        c_row = cons.get((row["llm"], row["run_index"]))
        return rollup([row["final_verdict"], c_row["final_verdict"] if c_row else "BLOCKED"])
    return row["final_verdict"]


def grade_test_cases(prompt_rows, cons_rows):
    """Roll prompt results up to the 5 test cases per app per run (NLC-02 = 02a AND 02b AND 02c AND consistent)."""
    cons = {(r["llm"], r["run_index"]): r for r in cons_rows}
    groups = defaultdict(list)
    for r in prompt_rows:
        if not r["augmented"]:
            groups[(r["llm"], r["test_case"], r["run_index"])].append(r)
    rows = []
    for (llm, tc, run), recs in sorted(groups.items()):
        verdicts = [r["final_verdict"] for r in recs]
        if tc == "NLC-02":
            c_row = cons.get((llm, run))
            verdicts.append(c_row["final_verdict"] if c_row else "BLOCKED")
            if len(recs) < 3:
                verdicts.append("BLOCKED")
        rows.append({"llm": llm, "test_case": tc, "run_index": run, "final_verdict": rollup(verdicts)})
    return rows


def grade_augmented(prompt_rows):
    """An augmented variant should get the same verdict as the prompt it was made from."""
    src = {(r["llm"], r["id"], r["run_index"]): r for r in prompt_rows if not r["augmented"]}
    rows = []
    for r in prompt_rows:
        if r["augmented"]:
            s = src.get((r["llm"], r["source_id"], r["run_index"]))
            rows.append({**r, "source_verdict": s["final_verdict"] if s else "NOT RUN",
                         "same_as_source": (r["final_verdict"] == s["final_verdict"]) if s else None})
    return rows


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
COLORS = {"PASS": "green", "FAIL": "red", "REVIEW": "yellow", "BLOCKED": "grey"}


def progress(msg):
    if sys.stderr.isatty():
        print(f"\r  {msg}".ljust(80), end="", file=sys.stderr, flush=True)


def clear_progress():
    if sys.stderr.isatty():
        print("\r" + " " * 80 + "\r", end="", file=sys.stderr, flush=True)


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def score(verdicts):
    """(passed, scored, review, blocked): BLOCKED is not scored; REVIEW is scored but not a pass."""
    cnt = Counter(verdicts)
    return cnt["PASS"], cnt["PASS"] + cnt["FAIL"] + cnt["REVIEW"], cnt["REVIEW"], cnt["BLOCKED"]


def score_cols(verdicts):
    p, s, rv, b = score(verdicts)
    return [f"{p}/{s}", pct(p, s), rv, b]


def print_table(title, headers, rows, note=""):
    """Print a boxed table (same look as Martin's report)."""
    rows = [[str(x) for x in r] for r in rows]
    w = [max(len(x) for x in col) for col in zip(headers, *rows)]
    line = lambda l, m, r: "  " + l + m.join("─" * (n + 2) for n in w) + r
    print("\n" + c(f"  {title}", "bold"))
    print(line("┌", "┬", "┐"))
    print("  │" + "│".join(f" {h.ljust(n)} " for h, n in zip(headers, w)) + "│")
    print(line("├", "┼", "┤"))
    for r in rows:
        print("  │" + "│".join(f" {x.ljust(n)} " for x, n in zip(r, w)) + "│")
    print(line("└", "┴", "┘"))
    if note:
        print(f"  {note}")


def ordered_llms(rows):
    names = {r["llm"] for r in rows}
    return [l for l in LLM_ORDER if l in names] + sorted(names - set(LLM_ORDER))


def print_main_results(tc_rows, runs, judge_txt):
    llms = ordered_llms(tc_rows)
    W = 70
    print("\n" + c("╔" + "═" * W + "╗", "cyan"))
    print(c("║", "cyan") + c("  NLC (NATURAL LANGUAGE CHATTING) TEST REPORT".ljust(W), "bold") + c("║", "cyan"))
    print(c("║", "cyan") + f"  Runs per test: {runs}    Grading: {judge_txt}"[:W].ljust(W) + c("║", "cyan"))
    print(c("╚" + "═" * W + "╝", "cyan"))

    # Results by test: PASS/FAIL when there is one run, "passes/runs" when there are several
    w0, wc = 8, 12
    line = lambda l, m, r: "  " + l + "─" * (w0 + 2) + "".join(m + "─" * (wc + 2) for _ in llms) + r
    print("\n" + c("  RESULTS BY TEST", "bold"))
    print(line("┌", "┬", "┐"))
    print("  │ " + "Test".ljust(w0) + " │" + "".join(" " + l.center(wc) + " │" for l in llms))
    print(line("├", "┼", "┤"))
    for tc in TEST_CASES:
        if not any(r["test_case"] == tc for r in tc_rows):
            continue
        out = f"  │ {tc.ljust(w0)} │"
        for l in llms:
            v = [r["final_verdict"] for r in tc_rows if r["test_case"] == tc and r["llm"] == l]
            if not v:
                text, color = "-", "grey"
            elif runs == 1:
                text, color = v[0], COLORS[v[0]]
            else:
                p = v.count("PASS")
                text = f"{p}/{len(v)}"
                color = "green" if p == len(v) else "red" if "FAIL" in v else "yellow"
            out += " " + c(text.center(wc), color) + " │"
        print(out)
    print(line("└", "┴", "┘"))

    print("\n" + c("  NLC SCORE", "bold"))
    for l in llms:
        p, s, rv, b = score([r["final_verdict"] for r in tc_rows if r["llm"] == l])
        pc = 100 * p / s if s else 0
        color = "green" if pc >= 80 else ("yellow" if pc >= 50 else "red")
        extra = f"  ({b} BLOCKED, not scored)" if b else ""
        print(f"  {l.ljust(11)} {bar(p, s)}  {c(f'{p}/{s}'.rjust(5), color)}  {c(f'{pc:.0f}%'.rjust(4), color)}{extra}")
    print("\n  " + c("PASS", "green") + " = met the pass criteria   " + c("FAIL", "red") + " = did not   "
          + c("REVIEW", "yellow") + " = the two checks disagree (not a pass)   "
          + c("BLOCKED", "grey") + " = no answer")


def print_details(prompt_rows, cons_rows, tc_rows, aug_rows, run_times, runs, judge_txt):
    base = [r for r in prompt_rows if not r["augmented"]]
    cons = {(r["llm"], r["run_index"]): r for r in cons_rows}
    llms = ordered_llms(prompt_rows)
    print("\n" + c("  ── DETAILED STATISTICS (for the report) " + "─" * 30, "cyan"))
    hdr = ["Passed", "Pass rate", "REVIEW", "BLOCKED"]

    print_table("BY APP (test-case results)", ["App"] + hdr,
                [[l] + score_cols([r["final_verdict"] for r in tc_rows if r["llm"] == l]) for l in llms] +
                [["All apps"] + score_cols([r["final_verdict"] for r in tc_rows])])

    eff = [(r, effective_verdict(r, cons)) for r in base]
    print_table("BY INPUT CLASS (prompt level, all apps and runs)", ["Input class", "Name", "Prompt"] + hdr,
                [[ic, IC[ic].split(" (")[0], ", ".join(sorted({r["id"] for r, _ in eff if r["input_class"] == ic}))]
                 + score_cols([v for r, v in eff if r["input_class"] == ic])
                 for ic in IC if any(r["input_class"] == ic for r, _ in eff)],
                note="NLC-02 prompts count as passed only if the answer passes AND the three wordings are consistent.")

    rule_rows = []
    for rule in sorted({r["rule"] for r in base}):
        ex = next(r for r in base if r["rule"] == rule)
        joiner = " or " if rule == "R6" else " + "
        rule_rows.append([rule, ex["context"], ex["input_class"], ex["id"], joiner.join(ex["expected_output_classes"])]
                         + score_cols([v for r, v in eff if r["rule"] == rule]))
    print_table("BY DECISION-TABLE RULE (prompt level)",
                ["Rule", "Who's asking", "Input class", "Prompt", "Expected output"] + hdr, rule_rows)

    observed = Counter(r["final_output_class"] for r in base + cons_rows if r["final_output_class"])
    print_table("OUTPUT CLASSES: DEFINED vs. OBSERVED", ["Output class", "Name", "Type", "Times observed"],
                [[k, v["name"], "Valid" if v["valid"] else "Invalid", observed.get(k, 0)] for k, v in OC.items()],
                note=f"Observed at least once: {sum(1 for k in OC if observed.get(k))}/{len(OC)}")

    cx_rows = []
    for lvl in ("Low", "Medium", "High"):
        tcs = [t for t in TEST_CASES if COMPLEXITY[t] == lvl]
        cx_rows.append([lvl, ", ".join(tcs)] + score_cols([r["final_verdict"] for r in tc_rows if r["test_case"] in tcs]))
    print_table("BY TEST COMPLEXITY (test-case results)", ["Complexity", "Test cases"] + hdr, cx_rows)

    if runs > 1:
        st_rows, n_stable = [], 0
        for tc in TEST_CASES:
            for l in llms:
                v = [r["final_verdict"] for r in sorted(tc_rows, key=lambda r: r["run_index"])
                     if r["test_case"] == tc and r["llm"] == l]
                if not v:
                    continue
                status = "Incomplete" if "BLOCKED" in v else "Stable" if len(set(v)) == 1 else "Unstable"
                n_stable += status == "Stable"
                st_rows.append([tc, l, " / ".join(v), f"{v.count('PASS')}/{len(v)}", status])
        print_table(f"STABILITY ACROSS {runs} RUNS", ["Test", "App", "Verdict per run", "Passes", "Stability"],
                    st_rows, note=f"Stable (same verdict every run): {n_stable}/{len(st_rows)}. "
                                  "Incomplete = a run got no answer (BLOCKED).")

    both = [r for r in prompt_rows + cons_rows
            if r["judge_result"] in ("PASS", "FAIL") and r["keyword_result"] in ("PASS", "FAIL")]
    agree = sum(r["keyword_result"] == r["judge_result"] for r in both)
    finals = [r["final_verdict"] for r in prompt_rows + cons_rows]
    print_table("GRADING SUMMARY", ["Measure", "Value"], [
        ["Grading", judge_txt],
        ["Answers graded (incl. NLC-02 consistency checks)", len(finals)],
        ["Keyword check and judge agreed", f"{agree}/{len(both)} ({pct(agree, len(both))})"],
        ["REVIEW (the two checks disagree)", finals.count("REVIEW")],
        ["BLOCKED (no answer from the app)", finals.count("BLOCKED")],
    ])

    t_rows = []
    for l in llms + [None]:
        rs = [r for r in prompt_rows if l is None or r["llm"] == l]
        resp = [r["response_seconds"] for r in rs if r.get("response_seconds")]
        wall = sum(t["seconds"] for t in run_times if l is None or t["llm"] == l)
        judge_s = sum(r["judge_seconds"] for r in rs + [x for x in cons_rows if l is None or x["llm"] == l])
        t_rows.append([l or "All apps", len(rs), f"{sum(resp) / len(resp):.1f}" if resp else "-",
                       f"{wall / 60:.1f}", f"{judge_s / 60:.1f}"])
    n_prompts = len({r["id"] for r in base})
    print_table("EXECUTIONS AND TIMING", ["App", "Prompts sent", "Avg answer time (s)", "Browser time (min)",
                                          "Judge time (min)"], t_rows,
                note=f"Base prompts: {n_prompts} x {runs} run(s) x {len(llms)} app(s) = "
                     f"{n_prompts * runs * len(llms)} planned; "
                     f"{sum(r['final_verdict'] != 'BLOCKED' for r in base)} answered. "
                     f"Test-case results: {len(tc_rows)}.")

    print_table("MEMORY OFF (how each app was run)", ["App", "Mode"],
                [[l, ", ".join(sorted({r["memory_mode"] for r in prompt_rows if r["llm"] == l}))] for l in llms])

    if aug_rows:
        a_rows = []
        for t in sorted({a["augmentation_type"] for a in aug_rows}):
            rs = [a for a in aug_rows if a["augmentation_type"] == t]
            same = [a for a in rs if a["same_as_source"] is not None]
            a_rows.append([t] + score_cols([a["final_verdict"] for a in rs]) +
                          [f"{sum(a['same_as_source'] for a in same)}/{len(same)}"])
        print_table("AUGMENTED VARIANTS (wording robustness)", ["Variant type"] + hdr + ["Same verdict as original"],
                    a_rows)
        for a in aug_rows:
            if a["same_as_source"] is False:
                print(c(f"  ! {a['llm']} {a['id']} run {a['run_index']}: {a['final_verdict']} but original "
                        f"{a['source_id']} was {a['source_verdict']}", "yellow"))

    # Anything a person needs to look at, with enough detail to decide on the spot
    issues = [r for r in prompt_rows + cons_rows if r["final_verdict"] in ("REVIEW", "BLOCKED", "FAIL")]
    if issues:
        print("\n" + c("  ITEMS TO LOOK AT", "bold"))
        for r in issues:
            print(c(f"\n  [{r['final_verdict']}] {r['llm']} {r['id']} run {r['run_index']}", COLORS[r["final_verdict"]]))
            print(f"    Keyword check: {r['keyword_result']} {r.get('keyword_output_class', '')} "
                  f"{r.get('keyword_explanation', '')}".rstrip())
            if r["judge_result"] != "SKIPPED":
                print(f"    Judge:         {r['judge_result']} {r.get('judge_output_class', '')} - {r['judge_reasoning']}")
            if r.get("actual_response"):
                print(f"    Answer starts: \"{' '.join(r['actual_response'].split())[:400]}...\"")
    print()


def grade_and_report(records, run_times, runs, use_judge=True, model="llama3.2:3b"):
    """Grade every answer and print the results. Called by nlc_test.py at the end of a run."""
    judge_txt = f"keyword check + local AI judge ({model})" if use_judge else "keyword check only"
    start = time.time()
    grade_prompts(records, model, use_judge)
    cons_rows = grade_consistency(records, model, use_judge)
    clear_progress()
    tc_rows = grade_test_cases(records, cons_rows)
    aug_rows = grade_augmented(records)
    print_main_results(tc_rows, runs, judge_txt)
    print_details(records, cons_rows, tc_rows, aug_rows, run_times, runs, judge_txt)
    print(f"  Grading took {time.time() - start:.0f}s. Nothing was saved - copy or screenshot this output if needed.\n")
