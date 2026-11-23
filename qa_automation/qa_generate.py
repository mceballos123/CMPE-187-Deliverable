#!/usr/bin/env python3
"""Test case generation, data discovery/setup and data augmentation
for the Question/Answering feature (CMPE 187 Deliverable 3).

  python qa_generate.py discover               # check test data, model and environment
  python qa_generate.py augment                # show every augmented input
  python qa_generate.py generate --plan 3d     # write generated/qa_suite_3d.json

The suite is built from the 3D test model in data/qa_model_3d.json:
context class x input class (with augmentation) x expected output class.
Augmentation is seeded, so the same seed always gives the same suite file.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
GEN = BASE / "generated"

DOMAIN_WORDS = {"snmp", "trap", "inform", "community", "counter", "interface", "table",
                "router", "switch", "octets", "manager", "agent", "string", "unit"}


def load(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------ augmentation

def add_noise(prompt, rng):
    """Casual, noisy user input: lowercase start, no final '?', two letter swaps.
    Tokens with digits, capitals, dots, dashes or colons are never touched."""
    parts = re.split(r"(\s+)", prompt)
    candidates = []
    for i, tok in enumerate(parts):
        core = tok.strip(",.?;:")
        if core.isalpha() and core.islower() and len(core) >= 5 and core not in DOMAIN_WORDS:
            candidates.append(i)
    for i in sorted(rng.sample(candidates, min(2, len(candidates)))):
        tok = parts[i]
        core = tok.strip(",.?;:")
        for _ in range(10):
            j = rng.randint(1, len(core) - 3)
            if core[j] != core[j + 1]:
                break
        swapped = core[:j] + core[j + 1] + core[j] + core[j + 2:]
        parts[i] = tok.replace(core, swapped, 1)
    text = "".join(parts).rstrip()
    text = text.rstrip("?").rstrip()
    if text[:1].isupper() and (len(text) < 2 or not text[1].isupper()):
        text = text[0].lower() + text[1:]
    return text


def param_variant(base, rng):
    """Return (prompt, oracle, key_points) with new values and a recomputed oracle."""
    pv = base["param_variant"]
    oracle = dict(base["oracle"])
    kind = pv["type"]
    if kind == "object_swap":
        c = rng.choice(pv["choices"])
        oracle.update(c)
        prompt = pv["template"].format(**c)
        kps = ["The OID is %s (instance %s.0)." % (c["oid"], c["oid"]),
               "The value is TimeTicks, counted in hundredths of a second."]
    elif kind == "counter_numbers":
        gap = rng.randint(*pv["first_gap_range"])
        first = 2 ** oracle.get("bits", 32) - gap
        second = rng.randint(*pv["second_range"])
        interval = rng.choice(pv["intervals"])
        oracle.update(first=first, second=second, interval=interval)
        delta = gap + second
        prompt = pv["template"].format(first="{:,}".format(first), second="{:,}".format(second),
                                       interval=interval)
        kps = ["Change = (4,294,967,296 - {:,}) + {:,} = {:,} octets.".format(first, second, delta),
               "{:,} x 8 = {:,} bits.".format(delta, delta * 8),
               "{:,} / {} = about {:,.0f} bps.".format(delta * 8, interval, delta * 8 / interval)]
    elif kind == "command_values":
        v = rng.choice(pv["versions"])
        ip = "10.%d.%d.%d" % (rng.randint(1, 254), rng.randint(0, 254), rng.randint(1, 254))
        community = "%s%d" % (rng.choice(pv["community_words"]), rng.randint(10, 99))
        oracle.update(ip=ip, community=community, version=v["version"])
        prompt = pv["template"].format(ip=ip, community=community, version_name=v["version_name"])
        kps = ["snmpwalk -v%s -c %s %s IF-MIB::ifTable" % (v["version"], community, ip),
               "The OID 1.3.6.1.2.1.2.2 is also accepted" +
               (", and so are snmpbulkwalk and snmptable." if v["version"] != "1"
                else ", and so is snmptable (snmpbulkwalk needs SNMPv2c).")]
    elif kind == "subject_swap":
        subject = rng.choice(pv["choices"])
        oracle.update(subject=subject)
        prompt = pv["template"].format(subject=subject)
        kps = ["%s does not exist; no RFC defines it." % subject,
               "SNMPv3 (RFC 3411 to 3418) is the latest version."]
    else:
        raise ValueError("unknown param_variant type " + kind)
    return prompt, oracle, kps


def build_inputs(model, data, seed=None):
    """Data augmentation: expand each base test case into ORIG/PARA/NOISE(/PARAM) inputs."""
    seed = data.get("seed", 187) if seed is None else seed
    rows = []
    for base in data["tests"]:
        rule = next(r for r in model["rules"] if r["rule"] == base["rule"])
        num = base["id"][-2:]
        # one RNG per base test, so adding a test never changes the others
        rng = random.Random("%s-%s" % (seed, base["id"]))
        common = {k: base[k] for k in ("rule", "input_class", "requirements", "complexity",
                                       "max_words", "pass_criteria")}
        common.update(base_id=base["id"], expected_output=rule["expected"],
                      failure_outputs=rule["failure"])
        variants = [("ORIG", base["prompt"], base["oracle"], base["reference_key_points"]),
                    ("PARA", base["paraphrase"], base["oracle"], base["reference_key_points"]),
                    ("NOISE", add_noise(base["prompt"], rng), base["oracle"], base["reference_key_points"])]
        if base.get("param_variant"):
            variants.append(("PARAM",) + param_variant(base, rng))
        for aug, prompt, oracle, kps in variants:
            row = dict(common)
            row.update(input_id="QA-%s-%s" % (num, aug), augmentation=aug, prompt=prompt,
                       oracle=oracle, reference_key_points=kps)
            rows.append(row)
    return rows


# --------------------------------------------------------- suite generation

def build_suite(model, data, plan="3d", seed=None):
    inputs = build_inputs(model, data, seed)
    ctx = model["context"]
    tests, seen = [], set()
    for cell in model["plans"][plan]["cells"]:
        for cx in cell["context"]:
            for row in inputs:
                if row["augmentation"] not in cell["augmentation"]:
                    continue
                test_id = "%s-%s" % (row["input_id"], cx.replace("QA-", "").replace("-", ""))
                if test_id in seen:
                    continue
                seen.add(test_id)
                t = dict(row)
                t.update(test_id=test_id, context_class=cx,
                         filler_turns=ctx["classes"][cx]["filler_turns"],
                         fillers=ctx["fillers"][:ctx["classes"][cx]["filler_turns"]])
                tests.append(t)
    tests.sort(key=lambda t: (t["base_id"], t["context_class"],
                              ["ORIG", "PARA", "NOISE", "PARAM"].index(t["augmentation"])))
    body = json.dumps(tests, sort_keys=True, ensure_ascii=False)
    meta = {
        "feature": model["feature"], "plan": plan, "plan_description": model["plans"][plan]["description"],
        "seed": data.get("seed", 187) if seed is None else seed,
        "apps": list(model["context"]["apps"]),
        "n_tests": len(tests), "n_executions_per_repeat": len(tests) * len(model["context"]["apps"]),
        "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    return {"meta": meta, "tests": tests}


def coverage_matrix(model, suite):
    ics = list(model["input"]["classes"])
    cxs = list(model["context"]["classes"])
    augs = ["ORIG", "PARA", "NOISE", "PARAM"]
    lines = ["%-10s %-22s" % ("Input", "Expected output") +
             "".join("%-22s" % cx for cx in cxs)]
    for ic in ics:
        rows = [t for t in suite["tests"] if t["input_class"] == ic]
        exp = rows[0]["expected_output"] if rows else "-"
        cells = []
        for cx in cxs:
            got = [t["augmentation"] for t in rows if t["context_class"] == cx]
            cells.append("%-22s" % (",".join(a for a in augs if a in got) or "-"))
        lines.append("%-10s %-22s" % (ic, exp) + "".join(cells))
    planned = len(ics) * len(cxs)
    filled = sum(1 for ic in ics for cx in cxs
                 if any(t["input_class"] == ic and t["context_class"] == cx for t in suite["tests"]))
    lines.append("")
    lines.append("Input x context cells covered: %d of %d. Apps per cell: %d. "
                 "Expected output classes used: %s." % (
                     filled, planned, len(model["context"]["apps"]),
                     ", ".join(sorted({t["expected_output"] for t in suite["tests"]}))))
    return "\n".join(lines)


# -------------------------------------------------------------- discovery

def validate(model, data):
    """Data discovery checks: everything the oracle and runner depend on is present and consistent."""
    problems = []
    rules = {r["rule"]: r for r in model["rules"]}
    oracle_types = set()
    spec = importlib.util.spec_from_file_location("qa_oracle", BASE / "qa_oracle.py")
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        oracle_types = set(mod.ORACLES)
    seen_ic, seen_req = set(), set()
    for t in data["tests"]:
        r = rules.get(t["rule"])
        if not r:
            problems.append("%s: rule %s is not in the model" % (t["id"], t["rule"]))
            continue
        if r["test"] != t["id"] or r["input"] != t["input_class"]:
            problems.append("%s: does not match decision rule %s" % (t["id"], t["rule"]))
        if t["input_class"] not in model["input"]["classes"]:
            problems.append("%s: unknown input class %s" % (t["id"], t["input_class"]))
        if r["expected"] not in model["output"]["classes"]:
            problems.append("%s: unknown expected output class" % t["id"])
        if t["oracle"]["type"] not in oracle_types:
            problems.append("%s: no oracle function for type %s" % (t["id"], t["oracle"]["type"]))
        if len(t["reference_key_points"]) < 2:
            problems.append("%s: fewer than 2 reference key points" % t["id"])
        for k in ("prompt", "paraphrase"):
            if not t.get(k, "").strip():
                problems.append("%s: empty %s" % (t["id"], k))
        seen_ic.add(t["input_class"])
        seen_req.update(t["requirements"])
    for ic in model["input"]["classes"]:
        if ic not in seen_ic:
            problems.append("input class %s has no test case" % ic)
    for req in model["requirements"]:
        if req not in seen_req:
            problems.append("requirement %s has no test case" % req)
    if len(model["context"]["fillers"]) < max(c["filler_turns"] for c in model["context"]["classes"].values()):
        problems.append("not enough filler prompts for the carry-over context")
    return problems


def find_chrome():
    for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "C:/Program Files/Google/Chrome/Application/chrome.exe"):
        if os.path.exists(p):
            return p
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def ollama_status(url="http://localhost:11434", model="llama3.2:3b"):
    try:
        import urllib.request
        with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        return "running, judge model %s %s" % (model, "pulled" if model in names else
                                               "NOT pulled (run: ollama pull %s)" % model)
    except Exception:
        return "not reachable at %s (judge will be skipped, oracle only)" % url


def cmd_discover(args):
    model, data = load("qa_model_3d.json"), load("qa_test_data.json")
    print("== Test data discovery")
    print("%-9s %-9s %-4s %-13s %-3s %-15s %s" % ("Test", "Input", "Rule", "Requirements", "KPs",
                                                 "Oracle", "PARAM variant"))
    for t in data["tests"]:
        print("%-9s %-9s %-4s %-13s %-3d %-15s %s" % (
            t["id"], t["input_class"], t["rule"], ",".join(t["requirements"]),
            len(t["reference_key_points"]), t["oracle"]["type"],
            (t.get("param_variant") or {}).get("type", "-")))
    print("\nModel: %d requirements, %d context classes x %d apps, %d input classes, %d output classes, %d rules"
          % (len(model["requirements"]), len(model["context"]["classes"]), len(model["context"]["apps"]),
             len(model["input"]["classes"]), len(model["output"]["classes"]), len(model["rules"])))
    problems = validate(model, data)
    print("Consistency checks: %s" % ("all passed" if not problems else "%d problem(s)" % len(problems)))
    for p in problems:
        print("  - " + p)

    print("\n== Environment setup")
    print("Python        %s (%s)" % (platform.python_version(), sys.executable))
    try:
        import selenium
        print("Selenium      %s" % selenium.__version__)
    except ImportError:
        print("Selenium      NOT installed (pip install -r requirements.txt)")
    print("Chrome        %s" % (find_chrome() or "NOT found"))
    prof = BASE / "chrome_profiles"
    for app in model["context"]["apps"]:
        p = prof / app
        print("Profile %-10s %s" % (app, "saved (log in once with --login if not signed in)" if p.exists()
                                     else "missing (run: python qa_runner.py --login --apps %s)" % app))
    print("Ollama judge  %s" % ollama_status())
    print("Results dir   %s" % (BASE / "results"))
    return 1 if problems else 0


def cmd_augment(args):
    model, data = load("qa_model_3d.json"), load("qa_test_data.json")
    rows = build_inputs(model, data, args.seed)
    GEN.mkdir(exist_ok=True)
    out = GEN / "qa_inputs.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print("%-14s %-6s %s" % (r["input_id"], r["augmentation"], r["prompt"]))
        if r["augmentation"] == "PARAM":
            print("%-21s expected: %s" % ("", " | ".join(r["reference_key_points"])))
    counts = {}
    for r in rows:
        counts[r["augmentation"]] = counts.get(r["augmentation"], 0) + 1
    print("\n%d base test cases -> %d inputs (%s). Written to %s" % (
        len(data["tests"]), len(rows), ", ".join("%s %d" % kv for kv in counts.items()), out))


def cmd_generate(args):
    model, data = load("qa_model_3d.json"), load("qa_test_data.json")
    problems = validate(model, data)
    if problems:
        print("Fix the test data first:\n  - " + "\n  - ".join(problems))
        return 1
    suite = build_suite(model, data, args.plan, args.seed)
    GEN.mkdir(exist_ok=True)
    out = GEN / ("qa_suite_%s.json" % args.plan)
    out.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    m = suite["meta"]
    print("Plan %s: %s" % (args.plan, m["plan_description"]))
    print("%d test cases x %d apps = %d executions per repeat" % (
        m["n_tests"], len(m["apps"]), m["n_executions_per_repeat"]))
    print("Seed %s, content sha256 %s" % (m["seed"], m["content_sha256"][:16]))
    print("Written to %s\n" % out)
    print(coverage_matrix(model, suite))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover", help="check test data, 3D model and environment")
    a = sub.add_parser("augment", help="run data augmentation and print every input")
    a.add_argument("--seed", type=int)
    g = sub.add_parser("generate", help="generate the 3D test suite")
    g.add_argument("--plan", choices=["core", "3d", "full"], default="3d")
    g.add_argument("--seed", type=int)
    args = ap.parse_args()
    return {"discover": cmd_discover, "augment": cmd_augment, "generate": cmd_generate}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
