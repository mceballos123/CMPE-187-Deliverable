#!/usr/bin/env python3
"""Test automation statistics for the Question/Answering runs.

  python qa_report.py results/run_XXXX [more run folders] [--dev-hours 10]
  python qa_report.py --complexity-only

Writes <first run>/report/: summary.md (all tables), one CSV per table, stats.json
and two charts. Everything is computed from results.jsonl (or results_regraded.jsonl),
review.jsonl (hand decisions) and timing.jsonl. Nothing is typed in by hand except the
cost assumptions given on the command line, which are printed with the numbers.
"""

import argparse
import ast
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import qa_generate
import qa_review

BASE = Path(__file__).resolve().parent
VERDICTS = ["Pass", "Partial Pass", "Fail", "REVIEW", "NOT RUN"]
SCRIPTS = ["qa_generate.py", "qa_oracle.py", "qa_judge.py", "qa_sites.py", "qa_runner.py",
           "qa_review.py", "qa_report.py"]
TESTS = ["tests/test_qa_oracle.py", "tests/test_qa_generate.py", "tests/test_qa_report.py"]


# ------------------------------------------------------------- complexity

class _McCabe(ast.NodeVisitor):
    def __init__(self):
        self.score = 1

    def generic_visit(self, node):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler,
                             ast.Assert, ast.comprehension)):
            self.score += 1
        if isinstance(node, ast.comprehension):
            self.score += len(node.ifs)
        if isinstance(node, ast.BoolOp):
            self.score += len(node.values) - 1
        if isinstance(node, ast.match_case):
            self.score += 1
        super().generic_visit(node)


def file_complexity(path):
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                doc_lines.update(range(first.lineno, first.end_lineno + 1))
    loc = sum(1 for i, ln in enumerate(lines, 1)
              if ln.strip() and not ln.strip().startswith("#") and i not in doc_lines)
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            v = _McCabe()
            v.visit(node)
            funcs.append((node.name, v.score))
    cc = [s for _n, s in funcs] or [0]
    top = max(funcs, key=lambda f: f[1]) if funcs else ("-", 0)
    return {"file": path.name, "total_lines": len(lines), "loc": loc, "functions": len(funcs),
            "avg_cc": round(sum(cc) / len(cc), 1), "max_cc": max(cc), "most_complex": "%s (%d)" % top}


def script_complexity():
    rows = [file_complexity(BASE / f) | {"role": "script"} for f in SCRIPTS if (BASE / f).exists()]
    rows += [file_complexity(BASE / f) | {"role": "unit test"} for f in TESTS if (BASE / f).exists()]
    data_files = ["data/qa_model_3d.json", "data/qa_test_data.json"]
    for f in data_files:
        p = BASE / f
        rows.append({"file": p.name, "role": "test data", "total_lines": len(p.read_text().splitlines()),
                     "loc": len(p.read_text().splitlines()), "functions": "-", "avg_cc": "-", "max_cc": "-",
                     "most_complex": "-"})
    return rows


def test_case_complexity(model, data):
    import qa_oracle
    inputs = {r["input_id"]: r for r in qa_generate.build_inputs(model, data)}
    out = []
    for t in data["tests"]:
        kind = t["oracle"]["type"]
        kps, _e, _f = qa_oracle.ORACLES[kind]("x", inputs["QA-%s-ORIG" % t["id"][-2:]]["oracle"])
        out.append({"test": t["id"], "input_class": t["input_class"], "level": t["complexity"],
                    "variants": 3 + (1 if t.get("param_variant") else 0),
                    "turns_cx01": 1, "turns_cx02": 1 + model["context"]["classes"]["QA-CX-02"]["filler_turns"],
                    "key_point_checks": len(kps), "error_checks": qa_oracle.ERROR_RULES[kind] + 1,
                    "oracle": kind})
    return out


# ------------------------------------------------------------- loading

def load_runs(run_dirs, use_regraded=True):
    rows, configs, timings = [], [], []
    for d in map(Path, run_dirs):
        name = "results_regraded.jsonl" if use_regraded and (d / "results_regraded.jsonl").exists() else "results.jsonl"
        reviews = qa_review.load_reviews(d)
        latest = {}
        for line in (d / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["auto_verdict"] = r.get("final_verdict")
            rv = reviews.get((r["app"], r["test_id"], r["repeat"]))
            if rv and r.get("final_verdict") == "REVIEW":
                r["final_verdict"] = rv["human_verdict"]
                r["human_reviewed"] = True
            latest[(r["app"], r["test_id"], r["repeat"])] = r  # a resumed retry replaces the earlier attempt
        rows += list(latest.values())
        configs.append(json.loads((d / "config.json").read_text(encoding="utf-8")))
        if (d / "timing.jsonl").exists():
            timings += [json.loads(x) for x in (d / "timing.jsonl").read_text().splitlines() if x.strip()]
    return rows, configs, timings


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def verdict_table(rows, key, order=None):
    groups = defaultdict(Counter)
    for r in rows:
        groups[key(r)][r["final_verdict"]] += 1
    keys = order or sorted(groups)
    out = []
    for k in keys:
        c = groups.get(k, Counter())
        executed = sum(c[v] for v in VERDICTS if v != "NOT RUN")
        out.append({"group": k, "executed": executed, "pass": c["Pass"], "partial": c["Partial Pass"],
                    "fail": c["Fail"], "review": c["REVIEW"], "not_run": c["NOT RUN"],
                    "pass_rate": pct(c["Pass"], executed),
                    "pass_or_partial": pct(c["Pass"] + c["Partial Pass"], executed)})
    total = Counter()
    for r in rows:
        total[r["final_verdict"]] += 1
    executed = sum(total[v] for v in VERDICTS if v != "NOT RUN")
    out.append({"group": "Total", "executed": executed, "pass": total["Pass"], "partial": total["Partial Pass"],
                "fail": total["Fail"], "review": total["REVIEW"], "not_run": total["NOT RUN"],
                "pass_rate": pct(total["Pass"], executed),
                "pass_or_partial": pct(total["Pass"] + total["Partial Pass"], executed)})
    return out


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


# ------------------------------------------------------------- main report

def oracle_of(r):
    return r.get("oracle") or {}


def why(r):
    o = oracle_of(r)
    reasons = o.get("errors", []) + o.get("fabrications", [])
    if not reasons:
        missing = [k["key_point"] for k in o.get("key_points", []) if not k["ok"]]
        reasons = ["missing: " + ", ".join(missing)] if missing else ["long answer (%s words)" % o.get("words")]
    return "; ".join(reasons)[:140]


def overview_table(rows, graded, configs, timings, apps, app_name):
    planned = sum(len(c["tests"]) * len(c["args"]["apps"]) * c["args"]["repeat"] for c in configs)
    status = Counter(r["status"] for r in rows)
    judge = sorted({(t.get("judge") or {}).get("status", "?") for t in timings})
    return [{
        "runs": ", ".join(c["run_id"] for c in configs),
        "plan": ", ".join(sorted({c["args"]["plan"] for c in configs})),
        "apps": ", ".join(app_name[a] for a in apps), "repeats": max(c["args"]["repeat"] for c in configs),
        "tests_in_plan": max(len(c["tests"]) for c in configs), "executions_planned": planned,
        "executions_recorded": len(rows), "graded": len(graded), "ok": status["ok"],
        "timeout": status["timeout"], "blocked": status["blocked"], "error": status["error"],
        "judge": ", ".join(judge) or "?",
        "wall_clock_min": round(sum(t["wall_secs"] for t in timings) / 60, 1)}]


NOT_RUN_REASONS = [
    ("challenge", "the site showed a human check (verify you are human) to the automated browser"),
    ("limit", "the site's message limit was reached"),
    ("login", "the site asked to sign in"),
    ("nosuchwindow", "the browser window was closed, so nothing could be typed"),
    ("window already closed", "the browser window was closed, so nothing could be typed"),
    ("something went wrong", "the site returned an error page"),
]


def not_run_reason(r):
    text = ("%s %s" % (r.get("error") or "", r.get("answer") or "")).lower()
    for needle, reason in NOT_RUN_REASONS:
        if needle in text:
            return reason
    return (r.get("error") or r.get("status") or "unknown").splitlines()[0][:80]


def not_run_table(rows, configs, apps, app_name):
    """Executions without a verdict, grouped by app and reason, plus planned ones that never started."""
    out = []
    for app in apps:
        app_rows = [r for r in rows if r["app"] == app]
        groups = Counter((r["status"], not_run_reason(r)) for r in app_rows if r["final_verdict"] == "NOT RUN")
        for (status, reason), n in groups.items():  # Counter keeps first-seen order, so the cause comes first
            out.append({"app": app_name[app], "status": status, "executions": n, "reason": reason})
        planned = sum(len(c["tests"]) * c["args"]["repeat"] for c in configs if app in c["args"]["apps"])
        if planned > len(app_rows):
            out.append({"app": app_name[app], "status": "not started", "executions": planned - len(app_rows),
                        "reason": "the runner stopped this app before these executions (see the rows above)"})
    return out


def test_app_matrix(graded, data, apps, app_name):
    out = []
    for t in data["tests"]:
        row = {"test": t["id"], "requirements": ",".join(t["requirements"])}
        tot_p = tot_n = 0
        for a in apps:
            rs = [r for r in graded if r["app"] == a and r["base_id"] == t["id"]]
            p = sum(r["final_verdict"] == "Pass" for r in rs)
            row[app_name[a]] = "%d/%d" % (p, len(rs))
            tot_p, tot_n = tot_p + p, tot_n + len(rs)
        row["total_pass"] = "%d/%d" % (tot_p, tot_n)
        row["pass_rate"] = pct(tot_p, tot_n)
        out.append(row)
    return out


def output_class_table(graded, model, apps, app_name):
    out = []
    for oc, info in model["output"]["classes"].items():
        row = {"output_class": oc, "name": info["name"], "verdict": info["verdict"]}
        for a in apps:
            row[app_name[a]] = sum(1 for r in graded if r["app"] == a and oracle_of(r).get("output_class") == oc)
        row["total"] = sum(row[app_name[a]] for a in apps)
        out.append(row)
    return out


def repeatability_tables(graded, apps, app_name):
    reps = defaultdict(list)
    for r in graded:
        reps[(r["app"], r["test_id"])].append((r["repeat"], r["final_verdict"]))
    multi = {k: v for k, v in reps.items() if len(v) > 1}
    stable = {k for k, v in multi.items() if len({verdict for _r, verdict in v}) == 1}
    rows, flaky = [], []
    for a in apps + [None]:
        keys = [k for k in multi if a is None or k[0] == a]
        same = [k for k in keys if k in stable]
        rows.append({"app": app_name[a] if a else "Total", "tests_with_2plus_repeats": len(keys),
                     "same_verdict_every_repeat": len(same), "stability": pct(len(same), len(keys))})
    for k in sorted(multi):
        if k not in stable:
            flaky.append({"app": app_name[k[0]], "test": k[1],
                          "verdicts": " / ".join(v for _r, v in sorted(multi[k]))})
    return rows, flaky


def validation_table(rows):
    judged = [r for r in rows if (r.get("judge") or {}).get("status") == "ok"]
    agree = sum(1 for r in judged if r["judge"]["verdict"] == r["oracle"]["verdict"])
    return [{
        "graded_by_oracle": sum(1 for r in rows if r.get("oracle")), "also_graded_by_judge": len(judged),
        "agreed": agree, "agreement_rate": pct(agree, len(judged)),
        "review_raised": sum(1 for r in rows if r.get("auto_verdict") == "REVIEW"),
        "review_decided_by_hand": sum(1 for r in rows if r.get("human_reviewed")),
        "review_still_open": sum(1 for r in rows if r["final_verdict"] == "REVIEW"),
        "body_diff_extractions": sum(1 for r in rows if r.get("extraction") == "body-diff"),
        "key_point_checks_run": sum(len(oracle_of(r).get("key_points", [])) for r in rows)}]


def _stat(xs, fn):
    return round(fn(xs), 1) if xs else "-"


def timing_table(graded, apps, app_name):
    out = []
    for a in apps + [None]:
        rs = [r for r in graded if a is None or r["app"] == a]
        rt = [r["response_secs"] for r in rs if r.get("response_secs") is not None]
        ex = [r["exec_secs"] for r in rs if r.get("exec_secs") is not None]
        words = [oracle_of(r).get("words", 0) for r in rs]
        out.append({"app": app_name[a] if a else "All", "executions": len(rs),
                    "mean_response_s": _stat(rt, statistics.mean), "median_response_s": _stat(rt, statistics.median),
                    "p90_response_s": _stat(rt, lambda x: quantile(x, 0.9)), "max_response_s": _stat(rt, max),
                    "mean_execution_s": _stat(ex, statistics.mean),
                    "sum_execution_min": _stat(ex, lambda x: sum(x) / 60),
                    "mean_words": _stat(words, statistics.mean)})
    return out


def cost_table(rows, configs, timings, review_raised, args):
    rows = [r for r in rows if r.get("status") in ("ok", "timeout")]  # executions that got an answer
    n_exec = len(rows)
    wall = sum(t["wall_secs"] for t in timings)
    seq_min = sum(r.get("exec_secs", 0) for r in rows) / 60
    judge_min = sum((r.get("judge") or {}).get("secs", 0) for r in rows) / 60
    manual_min = n_exec * args.manual_mins
    human_min = args.setup_mins * max(1, len(configs)) + review_raised * args.review_mins
    per_run_saving = (manual_min - human_min) / max(1, len(configs))
    scripts = [r for r in args.cx if r["role"] == "script"]
    tests = [r for r in args.cx if r["role"] == "unit test"]
    if args.dev_hours is not None and per_run_saving > 0:
        breakeven = "%.1f runs" % (args.dev_hours * 60 / per_run_saving)
    else:
        breakeven = "needs --dev-hours" if args.dev_hours is None else "no saving"
    return [
        {"item": "Development effort (scripts, data, oracle)",
         "value": ("%.1f h" % args.dev_hours) if args.dev_hours is not None else "fill in with --dev-hours",
         "basis": "own time log"},
        {"item": "Code written", "value": "%d LOC in %d scripts + %d LOC unit tests" % (
            sum(r["loc"] for r in scripts), len(scripts), sum(r["loc"] for r in tests)),
         "basis": "measured (non-blank, non-comment, no docstrings)"},
        {"item": "Executions that got an answer", "value": str(n_exec), "basis": "measured"},
        {"item": "Messages sent to chatbots", "value": str(sum(r.get("turns", 1) for r in rows)),
         "basis": "measured (test prompts + context fillers)"},
        {"item": "Machine time, wall clock", "value": "%.1f min" % (wall / 60), "basis": "measured (apps in parallel)"},
        {"item": "Machine time, sequential sum", "value": "%.1f min" % seq_min, "basis": "measured"},
        {"item": "Judge (Ollama) time", "value": "%.1f min" % judge_min, "basis": "measured, local model"},
        {"item": "Same work done by hand", "value": "%.0f min" % manual_min,
         "basis": "assumption: %.1f min per execution (type, wait, read, grade, record)" % args.manual_mins},
        {"item": "Human time with automation", "value": "%.0f min" % human_min,
         "basis": "assumption: %.0f min setup per run + %.1f min per REVIEW item" % (args.setup_mins, args.review_mins)},
        {"item": "Human time saved per run", "value": "%.0f min" % per_run_saving, "basis": "manual minus automated"},
        {"item": "Break-even point", "value": breakeven, "basis": "development time / time saved per run"},
        {"item": "Money spent", "value": "$%.2f" % args.dollars, "basis": "given with --dollars (plans, API)"},
    ]


def coverage_table(graded, configs, model, apps):
    def cov(item, done, total):
        return {"item": item, "covered": "%d of %d" % (done, total), "rate": pct(done, total)}
    planned_cells = {(a, t["context_class"], t["input_class"]) for c in configs for t in c["tests"]
                     for a in c["args"]["apps"]}
    done_cells = {(r["app"], r["context_class"], r["input_class"]) for r in graded}
    observed = sorted({oracle_of(r).get("output_class") for r in graded} - {None})
    planned = sum(len(c["tests"]) * len(c["args"]["apps"]) * c["args"]["repeat"] for c in configs)
    n_oc = len(model["output"]["classes"])
    full_space = len(model["context"]["apps"]) * len(model["context"]["classes"]) * len(model["input"]["classes"])
    return [
        cov("Requirements", len({q for r in graded for q in r["requirements"]}), len(model["requirements"])),
        cov("Decision rules", len({r["rule"] for r in graded}), len(model["rules"])),
        cov("Input classes", len({r["input_class"] for r in graded}), len(model["input"]["classes"])),
        cov("Context classes (session)", len({r["context_class"] for r in graded}), len(model["context"]["classes"])),
        cov("Apps", len({r["app"] for r in graded}), len(model["context"]["apps"])),
        cov("Augmentation types", len({r["augmentation"] for r in graded}), 4),
        {"item": "Expected output classes tested",
         "covered": ", ".join(sorted({t["expected_output"] for c in configs for t in c["tests"]})), "rate": "-"},
        {"item": "Output classes observed", "covered": "%d of %d (%s)" % (len(observed), n_oc, ", ".join(observed) or "none"),
         "rate": pct(len(observed), n_oc)},
        cov("3D cells (app x context x input) in the plan", len(done_cells & planned_cells), len(planned_cells)),
        cov("3D cells of the full model", len(done_cells), full_space),
        cov("Executions graded", len(graded), planned),
    ]


def build(run_dirs, args):
    model, data = qa_generate.load("qa_model_3d.json"), qa_generate.load("qa_test_data.json")
    rows, configs, timings = load_runs(run_dirs, not args.raw)
    app_name = dict(model["context"]["apps"])
    app_name[None] = "All"
    apps = [a for a in model["context"]["apps"] if any(r["app"] == a for r in rows)]
    graded = [r for r in rows if r["final_verdict"] != "NOT RUN"]
    T = {"run_overview": overview_table(rows, graded, configs, timings, apps, app_name),
         "verdict_by_app": verdict_table(rows, lambda r: app_name[r["app"]], [app_name[a] for a in apps]),
         "verdict_by_input_class": verdict_table(rows, lambda r: r["input_class"], list(model["input"]["classes"])),
         "verdict_by_context": verdict_table(rows, lambda r: r["context_class"], list(model["context"]["classes"])),
         "verdict_by_augmentation": verdict_table(rows, lambda r: r["augmentation"], ["ORIG", "PARA", "NOISE", "PARAM"]),
         "verdict_by_complexity": verdict_table(rows, lambda r: r["complexity"], ["Low", "Medium", "High"]),
         "pass_by_test_and_app": test_app_matrix(graded, data, apps, app_name),
         "observed_output_classes": output_class_table(graded, model, apps, app_name),
         "defects": [{"app": app_name[r["app"]], "test": r["test_id"], "repeat": r["repeat"],
                      "verdict": r["final_verdict"], "output_class": oracle_of(r).get("output_class"), "why": why(r)}
                     for r in graded if r["final_verdict"] in ("Fail", "Partial Pass")],
         "not_run": not_run_table(rows, configs, apps, app_name),
         "validation": validation_table(rows),
         "timing": timing_table(graded, apps, app_name),
         "coverage": coverage_table(graded, configs, model, apps),
         "script_complexity": args.cx,
         "test_case_complexity": test_case_complexity(model, data)}
    T["repeatability"], T["flaky_tests"] = repeatability_tables(graded, apps, app_name)
    T["automation_cost"] = cost_table(rows, configs, timings, T["validation"][0]["review_raised"], args)
    return T, rows, apps, app_name


def md_table(rows):
    if not rows:
        return "_none_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(c.replace("_", " ") for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def charts(T, rows, apps, app_name, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    status = {"Pass": "#0ca30c", "Partial Pass": "#fab219", "Fail": "#d03b3b", "REVIEW": "#9a9893"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "font.family": "DejaVu Sans"})

    # 1. verdicts per app (stacked horizontal bars)
    data = [r for r in T["verdict_by_app"] if r["group"] != "Total"]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(data) + 1.4), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    keys = [("pass", "Pass"), ("partial", "Partial Pass"), ("fail", "Fail"), ("review", "REVIEW")]
    for i, r in enumerate(data):
        left = 0
        for k, label in keys:
            v = r[k]
            if v:
                ax.barh(i, v, left=left, color=status[label], height=0.55, edgecolor=surface, linewidth=2,
                        label=label if i == 0 else None)
                if v / max(1, r["executed"]) > 0.08:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", color=ink if label != "Fail" else "white",
                            fontsize=9)
                left += v
        ax.text(left + 0.3, i, r["pass_rate"] + " pass", va="center", color=ink2, fontsize=9)
    ax.set_yticks(range(len(data)), [r["group"] for r in data], color=ink)
    ax.invert_yaxis()
    ax.set_xlabel("Executions")
    ax.xaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for _k, l in keys if l in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=9)
    ax.set_title("Final verdicts per app", loc="left", color=ink, fontsize=11, pad=26)
    fig.tight_layout()
    p = out / "chart_verdicts_by_app.png"
    fig.savefig(p, facecolor=surface)
    plt.close(fig)
    files.append(p.name)

    # 2. pass rate per input class, one series per app
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ics = [r["group"] for r in T["verdict_by_input_class"] if r["group"] != "Total"]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    w = 0.8 / max(1, len(apps))
    for j, a in enumerate(apps):
        vals = []
        for ic in ics:
            rs = [r for r in rows if r["app"] == a and r["input_class"] == ic and r["final_verdict"] != "NOT RUN"]
            vals.append(100.0 * sum(r["final_verdict"] == "Pass" for r in rs) / len(rs) if rs else 0)
        ax.bar([i + (j - (len(apps) - 1) / 2) * w for i in range(len(ics))], vals, width=w * 0.92,
               color=series[j % 4], label=app_name[a], edgecolor=surface, linewidth=1)
    ax.set_xticks(range(len(ics)), [ic.replace("QA-", "") for ic in ics])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate (%)")
    ax.yaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=len(apps), frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=9)
    ax.set_title("Pass rate per input class", loc="left", color=ink, fontsize=11, pad=26)
    fig.tight_layout()
    p = out / "chart_pass_rate_by_input_class.png"
    fig.savefig(p, facecolor=surface)
    plt.close(fig)
    files.append(p.name)
    return files


TITLES = [
    ("run_overview", "Run overview"),
    ("verdict_by_app", "Results per app"),
    ("not_run", "Executions not run, and why"),
    ("pass_by_test_and_app", "Passes per test case and app (passed / executed)"),
    ("verdict_by_input_class", "Results per input class (3D model: input)"),
    ("verdict_by_context", "Results per context class (3D model: context)"),
    ("observed_output_classes", "Observed output classes (3D model: output)"),
    ("verdict_by_augmentation", "Results per augmentation type"),
    ("verdict_by_complexity", "Results per test complexity level"),
    ("repeatability", "Repeatability: same verdict on every repeat"),
    ("flaky_tests", "Tests whose verdict changed between repeats"),
    ("validation", "Automatic validation"),
    ("defects", "Defects found (Fail and Partial Pass)"),
    ("timing", "Execution time"),
    ("automation_cost", "Test automation cost"),
    ("script_complexity", "Test script complexity"),
    ("test_case_complexity", "Test case complexity"),
    ("coverage", "Test coverage"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="results/<run_id> folders")
    ap.add_argument("--out", help="output folder (default <first run>/report)")
    ap.add_argument("--raw", action="store_true", help="ignore results_regraded.jsonl")
    ap.add_argument("--dev-hours", type=float, help="hours you spent building the automation")
    ap.add_argument("--manual-mins", type=float, default=3.0, help="minutes to run and grade one test by hand")
    ap.add_argument("--setup-mins", type=float, default=15.0, help="minutes of setup per automated run")
    ap.add_argument("--review-mins", type=float, default=1.5, help="minutes to decide one REVIEW item")
    ap.add_argument("--dollars", type=float, default=0.0, help="money spent on the run")
    ap.add_argument("--complexity-only", action="store_true")
    args = ap.parse_args(argv)
    args.cx = script_complexity()
    if args.complexity_only or not args.runs:
        model, data = qa_generate.load("qa_model_3d.json"), qa_generate.load("qa_test_data.json")
        print("## Test script complexity\n\n" + md_table(args.cx))
        print("## Test case complexity\n\n" + md_table(test_case_complexity(model, data)))
        return 0
    T, rows, apps, app_name = build(args.runs, args)
    out = Path(args.out) if args.out else Path(args.runs[0]) / "report"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Question/Answering test automation results\n",
          "Generated by qa_report.py from %s.\n" % ", ".join(Path(r).name for r in args.runs)]
    for key, title in TITLES:
        md.append("## %s\n\n%s" % (title, md_table(T[key])))
        if T[key]:
            with open(out / ("%s.csv" % key), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(T[key][0].keys()))
                w.writeheader()
                w.writerows(T[key])
    imgs = charts(T, rows, apps, app_name, out)
    for img in imgs:
        md.append("![%s](%s)\n" % (img, img))
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(T, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md[:2]))
    print("## Results per app\n\n" + md_table(T["verdict_by_app"]))
    print("Report written to %s (summary.md, %d CSV tables, %d charts, stats.json)" % (out, len(TITLES), len(imgs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
