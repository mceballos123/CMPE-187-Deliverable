#!/usr/bin/env python3
"""
Test-data augmentation: make new wordings of the NLC prompts, checked by a person before use.

Step 1 - generate candidates (uses the local Ollama model):
    python3 NLC/nlc_augment.py                         # 1 variant of each type for every base prompt
    python3 NLC/nlc_augment.py --tests NLC-03 --per-type 3
  Writes nlc_augmented_candidates.json. Every candidate starts as "status": "pending".

Step 2 - a person reviews the file: set "status" to "approved" or "rejected" for each candidate
  (the prompt text may be edited before approving). A variant must keep the same meaning,
  input class, expected result and pass criteria as the prompt it came from.

Step 3 - merge the approved ones into the test data:
    python3 NLC/nlc_augment.py --merge
  Approved variants are added to nlc_test_data.json with "augmented": true and "source_id".
  Run them with:  python3 NLC/nlc_test.py --include-augmented

    python3 NLC/nlc_augment.py --status                # just print generated / approved / rejected counts
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from nlc_evaluate import ask_ollama
from nlc_test import DATA_FILE

BASE_DIR = Path(__file__).resolve().parent
CANDIDATES_FILE = BASE_DIR / "nlc_augmented_candidates.json"

# The three kinds of variant. They target the team's documented gaps ("no heavy typos",
# "English-style only") and match the wording variants used in the IM tests.
TYPES = {
    "paraphrase": ("P", "Reword the message using different words and sentence structure. Keep the same tone and "
                        "the same level of formality as the original."),
    "heavy-typo": ("T", "Keep roughly the same words but type it like someone typing fast on a phone: many spelling "
                        "mistakes, swapped or missing letters, missing punctuation and capitals. A person must still "
                        "be able to understand it."),
    "non-native": ("N", "Rewrite it the way a non-native English speaker or a very casual texter might write it: "
                        "simple words, small grammar mistakes, abbreviations like 'u', 'pls', 'idk', 'cant'."),
}

# What each variant must keep, so it stays in the same input class with the same expected result.
KEEP = {
    "NLC-01": "It is a clear question from a new IT intern whose manager wants to know when switches or routers go "
              "down without walking to each one. Do NOT mention SNMP, monitoring software, or any tool name.",
    "NLC-02a": "It is a formal, technical question asking what security improvements SNMPv3 has over SNMPv2c.",
    "NLC-02b": "It is a casual question asking why people say SNMP v2c is not secure but v3 is, and what is different.",
    "NLC-02c": "It is an indirect question: the user's manager says to stop using community strings for SNMP on the "
               "routers; the user asks what to switch to and what it does better. Do NOT name SNMPv3.",
    "NLC-03": "It must stay AMBIGUOUS: the user only says their network monitoring stopped working and asks how to "
              "fix it. Do NOT add which tool, what the symptoms are, or what changed.",
    "NLC-04": "It must stay a short FRAGMENT, not a full sentence: SNMP stopped working after the community string "
              "was changed. Do NOT add any other details.",
    "NLC-05": "Keep EVERY detail: around 9 every morning the monitoring dashboard shows 6 or 7 switches as down for "
              "a few minutes, then green again; nobody complains about the network; a coworker thinks the "
              "switches are rebooting; the user checked and their uptime is fine; asks what is going on.",
}

# A formal prompt that becomes casual is no longer NL-IC-02 (formal wording), so it gets no non-native variant.
SKIP = {("NLC-02a", "non-native")}

PROMPT = """You help create test data for testing chatbots. Rewrite the user message below.

ORIGINAL MESSAGE:
{prompt}

HOW TO REWRITE IT: {how}

WHAT MUST STAY THE SAME: {keep} The rewritten message must mean the same thing and ask for the same help.
Do not answer the message. Do not add new facts.

Write {n} different rewritten versions. Output ONLY valid JSON:
{"variants": ["first version", "second version"]}"""


def load_data():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def load_candidates():
    return json.loads(CANDIDATES_FILE.read_text(encoding="utf-8")) if CANDIDATES_FILE.exists() else []


def save_candidates(cands):
    CANDIDATES_FILE.write_text(json.dumps(cands, indent=2, ensure_ascii=False), encoding="utf-8")


def generate(args):
    base = [p for p in load_data()["prompts"] if not p["augmented"]]
    if args.tests:
        base = [p for p in base if p["id"] in args.tests or p["test_case"] in args.tests]
    cands = load_candidates()
    existing = {c["prompt"].strip().lower() for c in cands}
    added = 0
    for p in base:
        for t in args.types:
            if (p["id"], t) in SKIP:
                continue
            print(f"  {p['id']:<8} {t:<11} ...", end=" ", flush=True)
            prompt = (PROMPT.replace("{prompt}", p["prompt"]).replace("{how}", TYPES[t][1])
                      .replace("{keep}", KEEP[p["id"]]).replace("{n}", str(args.per_type)))
            try:
                variants = ask_ollama(prompt, args.model, temperature=0.8).get("variants", [])
            except Exception as e:
                sys.exit(f"\nCould not reach Ollama ({type(e).__name__}). Is the Ollama app running?")
            new = 0
            for v in variants[:args.per_type]:
                v = str(v).strip()
                if not v or v.lower() in existing or v.lower() == p["prompt"].lower():
                    continue
                existing.add(v.lower())
                n = sum(c["source_id"] == p["id"] and c["augmentation_type"] == t for c in cands) + 1
                cands.append({"candidate_id": f"{p['id']}-{TYPES[t][0]}{n}", "source_id": p["id"],
                              "test_case": p["test_case"], "augmentation_type": t, "prompt": v,
                              "status": "pending", "note": "", "model": args.model,
                              "generated": datetime.now().isoformat(timespec="seconds")})
                new += 1
            added += new
            print(f"{new} new")
            save_candidates(cands)
    print(f"\n{added} candidate(s) added to {CANDIDATES_FILE.name}.")
    print('Next: open the file, set each "status" to "approved" or "rejected", then run:  '
          'python3 NLC/nlc_augment.py --merge')


def merge():
    data, cands = load_data(), load_candidates()
    by_id = {p["id"]: p for p in data["prompts"]}
    merged = 0
    for c in cands:
        if c["status"] != "approved" or c["candidate_id"] in by_id:
            continue
        item = dict(by_id[c["source_id"]])          # copy rule, classes, expected result, pass criteria
        item.update({"id": c["candidate_id"], "prompt": c["prompt"], "consistency_group": None,
                     "augmented": True, "source_id": c["source_id"], "augmentation_type": c["augmentation_type"]})
        data["prompts"].append(item)
        by_id[item["id"]] = item
        merged += 1
    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged {merged} newly approved variant(s) into {DATA_FILE.name}.")
    status()


def status():
    cands = load_candidates()
    count = lambda s, t=None: sum(c["status"] == s and (t is None or c["augmentation_type"] == t) for c in cands)
    print(f"\n  {'Type':<12}{'Generated':>10}{'Approved':>10}{'Rejected':>10}{'Pending':>9}")
    for t in TYPES:
        n = sum(c["augmentation_type"] == t for c in cands)
        print(f"  {t:<12}{n:>10}{count('approved', t):>10}{count('rejected', t):>10}{count('pending', t):>9}")
    print(f"  {'Total':<12}{len(cands):>10}{count('approved'):>10}{count('rejected'):>10}{count('pending'):>9}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge", action="store_true", help="Add approved candidates to nlc_test_data.json")
    ap.add_argument("--status", action="store_true", help="Print generated / approved / rejected counts")
    ap.add_argument("--tests", nargs="+", help="Only these prompts, e.g. NLC-03 NLC-02 (= 02a, 02b, 02c)")
    ap.add_argument("--types", nargs="+", default=list(TYPES), choices=list(TYPES))
    ap.add_argument("--per-type", type=int, default=1, help="Variants to generate per prompt per type")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model used to write the variants")
    args = ap.parse_args()
    if args.merge:
        merge()
    elif args.status:
        status()
    else:
        generate(args)


if __name__ == "__main__":
    main()
