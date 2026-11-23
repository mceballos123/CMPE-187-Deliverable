#!/usr/bin/env python3
"""
NLC (Natural Language Chatting) test runner for ChatGPT, Claude, Gemini and Perplexity.

One command does everything, for every app -> every run -> every prompt:
  1. opens a brand-new TEMPORARY / INCOGNITO chat (so the app has no memory of other prompts)
  2. types the prompt from nlc_test_data.json and presses Enter
  3. waits until the answer has finished appearing
  4. grades the answer with a keyword check (stage 1)
At the end, a local AI judge double-checks every answer (stage 2, see nlc_evaluate.py) and the
results table + statistics are printed in the terminal. Nothing is saved: every run starts fresh.

Usage:
    python3 NLC/nlc_test.py                                   # all apps, all prompts, 1 run
    python3 NLC/nlc_test.py --llms ChatGPT --tests NLC-01     # one app, one test
    python3 NLC/nlc_test.py --tests NLC-02                    # NLC-02 = all three wordings (02a, 02b, 02c)
    python3 NLC/nlc_test.py --runs 3                          # repeat everything 3 times
    python3 NLC/nlc_test.py --include-augmented               # also run approved augmented variants
    python3 NLC/nlc_test.py --no-judge                        # keyword check only (Ollama not needed)
    python3 NLC/nlc_test.py --login --llms Claude             # setup: log in once (close the window when done)

The browser-control functions (open chat, send prompt, wait for answer) are reused from
llm_memory_test.py; that file is not modified.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from selenium.webdriver.common.by import By

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # Martin's files are one folder up
from llm_memory_test import (
    SITES, make_driver, find_all_responses, safe_text, open_new_chat, send_prompt, wait_for_response,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "nlc_test_data.json"

# --------------------------------------------------------------------------- #
# Site settings: Martin's SITES, with NLC-only fixes where a site has changed.
# (These fixes are kept here so llm_memory_test.py is not modified.)
# --------------------------------------------------------------------------- #
NLC_SITES = {name: dict(cfg) for name, cfg in SITES.items()}
# ChatGPT (logged in) no longer has #prompt-textarea; the chat box is a ProseMirror editor.
NLC_SITES["ChatGPT"]["input"] = ["div.ProseMirror[contenteditable='true']",
                                 "div[contenteditable='true'][role='textbox']"] + SITES["ChatGPT"]["input"]
NLC_SITES["ChatGPT"]["response"] = ["div[class*='DilResponseRoot']"] + SITES["ChatGPT"]["response"]

# --------------------------------------------------------------------------- #
# Temporary / incognito chat settings ("memory off"), one per app.
#   temp_url : a web address that opens a temporary chat directly
#   clicks   : buttons to click after the page opens, in order. Each step is a list of
#              XPaths tried in order; if any step finds nothing, the prompt is recorded as
#              ERROR (BLOCKED) instead of quietly running with memory on.
# --------------------------------------------------------------------------- #
TEMP_CHAT = {
    # ChatGPT's temporary chat can still use memory, so also switch it to "Unpersonalized"
    # ("Ignores memory, plugins, and custom instructions").
    "ChatGPT": {"temp_url": "https://chatgpt.com/?temporary-chat=true",
                "clicks": [["//button[contains(normalize-space(), 'ersonaliz')]"],
                           ["//*[@role='menuitemradio'][contains(normalize-space(), 'Unpersonalized')]"]],
                "memory_mode": "Temporary chat + Unpersonalized (URL + menu)"},
    "Claude": {"temp_url": "https://claude.ai/new?incognito",
               "memory_mode": "Incognito chat (URL)"},
    "Gemini": {"clicks": [["//button[contains(@aria-label, 'emporary chat')]",
                           "//*[@role='button' or self::button][normalize-space()='Temporary chat']"]],
               "memory_mode": "Temporary chat (button)"},
    "Perplexity": {"clicks": [["//button[contains(@aria-label, 'ncognito')]",
                               "//*[@role='button' or self::button][contains(normalize-space(), 'ncognito')]"]],
                   "memory_mode": "Incognito (button)"},
}


def open_temp_chat(driver, llm):
    """Open a fresh chat with memory off. Returns the memory mode used (text for the report)."""
    cfg = dict(NLC_SITES[llm])
    temp = TEMP_CHAT[llm]
    if "temp_url" in temp:
        cfg["url"] = temp["temp_url"]          # Martin's open_new_chat loads cfg["url"]
    open_new_chat(driver, cfg)
    if is_logged_out(driver):                  # temporary chat needs an account; logged out = no memory anyway
        return "Logged out (no saved memory)"
    for step in temp.get("clicks", []):
        deadline = time.time() + 10            # buttons can appear a moment after the page loads
        clicked = click_first(driver, step)
        while not clicked and time.time() < deadline:
            time.sleep(0.5)
            clicked = click_first(driver, step)
        if not clicked:
            raise RuntimeError(f"Could not switch {llm} to temporary/incognito mode (button not found)")
        time.sleep(1.0)                        # let the page/menu react
    return temp["memory_mode"]


def is_logged_out(driver):
    """True if the page shows a 'Log in' / 'Sign in' button (the user is not signed in)."""
    xp = ("//*[self::button or self::a][normalize-space()='Log in' or normalize-space()='Sign in' "
          "or normalize-space()='Sign In']")
    return any(el.is_displayed() for el in driver.find_elements(By.XPATH, xp))


def click_first(driver, xpaths):
    for xp in xpaths:
        for el in driver.find_elements(By.XPATH, xp):
            try:
                if el.is_displayed():
                    el.click()
                    return True
            except Exception:
                continue
    return False


# --------------------------------------------------------------------------- #
# Keyword checks (stage 1 of grading).
# Each returns (passed, output_class, explanation). They are simple word/phrase
# searches that mirror the pass criteria; the AI judge in nlc_evaluate.py double-checks.
# --------------------------------------------------------------------------- #
def has(text, pattern):
    return re.search(pattern, text, re.I) is not None


def count_matches(text, patterns):
    return sum(has(text, p) for p in patterns)


def check_nlc01(r):
    monitoring = (r"\bsnmp\b|network monitoring|monitoring (tool|system|software|platform|solution)|\bnms\b|"
                  r"network management system|nagios|zabbix|prtg|librenms|solarwinds|observium|uptime kuma")
    if not has(r, monitoring):
        return False, "NL-OC-06", "Does not point to network monitoring, an NMS, or SNMP."
    note = ""
    words = len(r.split())
    if words > 600:  # NL-08: flag (do not fail) answers that may be too long for a beginner
        note = f"Note: long answer ({words} words) for a beginner (NL-08)."
    return True, "NL-OC-01", note


# The four core facts every NLC-02 answer must contain.
NLC02_FACTS = {
    "community strings (v2c)": r"community string",
    "sent in plain text (v2c)": r"plain[- ]?text|clear[- ]?text|unencrypted|not encrypted|without encryption|"
                                r"no encryption|in the clear",
    "authentication (v3)": r"authenticat|\busm\b|user[- ]based security",
    "encryption (v3)": r"\bencrypt(s|ed|ion)?\b|\bprivacy\b|\baes\b|\bdes\b",
}


def nlc02_facts(r):
    """Which of the four core facts the answer mentions (used for the consistency check too)."""
    return sorted(name for name, pattern in NLC02_FACTS.items() if has(r, pattern))


def check_nlc02(r, prompt_id=""):
    missing = [name for name in NLC02_FACTS if name not in nlc02_facts(r)]
    if missing:
        return False, "NL-OC-06", "Missing core fact(s): " + ", ".join(missing) + "."
    if prompt_id == "NLC-02c" and not has(r, r"snmp\s*-?\s*v?3|version 3"):
        return False, "NL-OC-06", "Does not connect 'stop using community strings' to SNMPv3."
    return True, "NL-OC-01", ""


def check_nlc03(r):
    asks = (r"(which|what) (monitoring )?(tool|software|system|platform|product)[^.?\n]*\?|"
            r"what (changed|symptoms?|error|exactly)[^.?\n]*\?|"
            r"(could|can) you (tell|share|provide|clarify|describe|give)|let me know (which|what|how|when|if you)|"
            r"(are|do) you (using|seeing|see|get|getting|have)[^.?\n]*\?|"
            r"(share|give|provide|tell me) (me )?(more|a few|some) (details|information|info)")
    hedges = (r"depend|could be|might be|may be|several (possible|common|things)|a few (possible|common|things)|"
              r"many (possible )?(causes|reasons|things)|without (more|knowing)|assum|(likely|common|possible) causes|"
              r"narrow (it|this|things) down|it'?s hard to say")
    causes = [r"service|process|crash", r"credential|password|community string", r"firewall|port",
              r"disk|storage|database", r"licen[cs]e", r"certificate|ssl|tls", r"update|upgrade|patch",
              r"network (path|connectivity)|unreachable|dns", r"agent", r"snmp|polling"]
    if has(r, asks):
        return True, "NL-OC-03", ""
    if has(r, hedges) and count_matches(r, causes) >= 2:
        return True, "NL-OC-04", ""
    return False, "NL-OC-08", "No clarifying question and no stated assumptions / multiple possible causes."


def check_nlc04(r):
    if not has(r, r"community string"):
        return False, "NL-OC-06", "Does not mention the community string."
    mismatch = (r"mismatch|must match|needs? to match|(don'?t|do not|doesn'?t|does not|no longer) match|"
                r"(update|change)[^.\n]{0,80}(nms|monitoring|manager|poller|server|tool|software|side)|"
                r"(nms|monitoring|manager|poller)[^.\n]{0,80}(old|previous) (community|string)|still using the old|"
                r"same (community )?string on both")
    if not has(r, mismatch):
        return False, "NL-OC-06", "Does not name a community string mismatch between the device and the NMS."
    return True, "NL-OC-01", ""


def check_nlc05(r):
    agrees_reboot = re.compile(
        r"(switch(es)?|they|devices?) (are|were|is|might be|may be) (probably |likely |actually |really )?"
        r"(rebooting|restarting|going down|crashing)|coworker (is|was|may be|might be) (right|correct)", re.I)
    negation = re.compile(r"\b(not|no|never|unlikely)\b|n't|rather than|instead of", re.I)
    for sentence in re.split(r"(?<=[.!?\n])\s+", r):
        if agrees_reboot.search(sentence) and not negation.search(sentence):
            return False, "NL-OC-09", f"Goes along with the reboot theory: \"{sentence.strip()[:160]}\""
    monitoring_side = (r"\bpoll|time[ds]? ?out|missed (responses?|polls?)|(not|n't) (getting|receiving|get) "
                       r"(a )?respon|no respon|monitoring (side|system|server|tool|issue|problem)|"
                       r"false (positive|alarm|alert)|dropped (packets|polls|requests)|icmp|ping")
    cause = (r"time[ds]? ?out|retr(y|ies)|\bcpu\b|\bload\b|traffic|congest|backup|scheduled|cron|"
             r"spanning[- ]tree|\bstp\b|rate[- ]limit|control[- ]plane|copp|login|logon|bandwidth|busy")
    if not has(r, monitoring_side):
        return False, "NL-OC-06", "Does not identify a monitoring/polling problem."
    if not has(r, cause):
        return False, "NL-OC-06", "Identifies a monitoring problem but suggests no relevant cause."
    return True, "NL-OC-05", ""


CHECKS = {"NLC-01": check_nlc01, "NLC-02": check_nlc02, "NLC-03": check_nlc03,
          "NLC-04": check_nlc04, "NLC-05": check_nlc05}


def keyword_check(item, response):
    """Run the right check for a prompt (augmented variants use their test case's check)."""
    if item["test_case"] == "NLC-02":
        return check_nlc02(response, item.get("source_id") or item["id"])
    return CHECKS[item["test_case"]](response)


# --------------------------------------------------------------------------- #
# Loading test data
# --------------------------------------------------------------------------- #
def load_prompts(selected=None, include_augmented=False):
    """Pick prompts. 'NLC-02' selects 02a, 02b and 02c; 'NLC-02b' selects just that one."""
    prompts = json.loads(DATA_FILE.read_text(encoding="utf-8"))["prompts"]
    if not include_augmented:
        prompts = [p for p in prompts if not p["augmented"]]
    if selected:
        known = {p["id"] for p in prompts} | {p["test_case"] for p in prompts}
        unknown = set(selected) - known
        if unknown:
            sys.exit(f"Unknown test IDs: {', '.join(sorted(unknown))}")
        prompts = [p for p in prompts if p["id"] in selected or p["test_case"] in selected
                   or p.get("source_id") in selected]
    return prompts


# --------------------------------------------------------------------------- #
# Running one prompt
# --------------------------------------------------------------------------- #
def run_prompt(driver, llm, item, run_index, timeout):
    """Send one prompt in a fresh temporary chat and return a record with the answer and keyword grade."""
    record = dict(item)
    record.update({"llm": llm, "run_index": run_index, "memory_mode": TEMP_CHAT[llm]["memory_mode"],
                   "actual_response": "", "keyword_result": "ERROR", "keyword_output_class": "",
                   "keyword_explanation": "", "response_seconds": None})
    cfg = NLC_SITES[llm]
    try:
        record["memory_mode"] = open_temp_chat(driver, llm)
        before = find_all_responses(driver, cfg["response"])
        prev, prev_text = len(before), (safe_text(before[-1]) if before else "")
        sent = time.time()
        send_prompt(driver, cfg, item["prompt"])
        record["actual_response"] = wait_for_response(driver, cfg, prev, timeout, prev_text=prev_text)
        record["response_seconds"] = round(time.time() - sent, 1)
        passed, oc, why = keyword_check(item, record["actual_response"])
        record.update({"keyword_result": "PASS" if passed else "FAIL",
                       "keyword_output_class": oc, "keyword_explanation": why})
    except Exception as e:  # no answer / site problem -> ERROR (reported as BLOCKED)
        record["keyword_explanation"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    return record


def login(llms):
    """Open each app's saved browser profile so the user can log in once; logins are remembered."""
    for llm in llms:
        print(f"\n=== {llm} ===  Log in in the Chrome window, then CLOSE that window to continue.", flush=True)
        driver = make_driver(llm)
        driver.get(SITES[llm]["url"])
        try:
            while driver.window_handles:   # wait until the user closes the window
                time.sleep(1)
        except Exception:
            pass
        finally:
            try:
                driver.quit()
            except Exception:
                pass
    print("\nLogins saved. You can now run the tests.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llms", nargs="+", default=list(SITES), choices=list(SITES), metavar="LLM",
                    help=f"Subset of: {', '.join(SITES)}")
    ap.add_argument("--tests", nargs="+", help="Test IDs, e.g. NLC-01 NLC-02 (= 02a+02b+02c) NLC-02b")
    ap.add_argument("--runs", type=int, default=1, help="How many times to repeat every prompt (default 1)")
    ap.add_argument("--include-augmented", action="store_true", help="Also run approved augmented variants")
    ap.add_argument("--no-judge", action="store_true", help="Skip the local AI judge (keyword check only)")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model used as the judge")
    ap.add_argument("--timeout", type=int, default=180, help="Max seconds to wait per response")
    ap.add_argument("--review", action="store_true", help="Pause after each prompt until you press Enter")
    ap.add_argument("--login", action="store_true",
                    help="Setup only: open each app's test browser so you can log in; close the window when done")
    args = ap.parse_args()
    if args.login:
        return login(args.llms)

    from nlc_evaluate import grade_and_report, ollama_running
    if not args.no_judge and not ollama_running():   # check before spending time in the browser
        sys.exit("Ollama is not running. Open the Ollama app, or run without the judge:  "
                 "python3 NLC/nlc_test.py --no-judge")

    prompts = load_prompts(args.tests, args.include_augmented)
    total = len(args.llms) * args.runs * len(prompts)
    print(f"{len(prompts)} prompt(s) x {args.runs} run(s) x {len(args.llms)} app(s) = {total} prompt executions")

    results, run_times = [], []
    for llm in args.llms:
        print(f"\n=== {llm} ({TEMP_CHAT[llm]['memory_mode']}) ===")
        driver = make_driver(llm)
        try:
            for run_index in range(1, args.runs + 1):
                run_start = time.time()
                for item in prompts:
                    print(f"  run {run_index}  {item['id']:<12}...", end=" ", flush=True)
                    rec = run_prompt(driver, llm, item, run_index, args.timeout)
                    results.append(rec)
                    secs = f"{rec['response_seconds']}s" if rec["response_seconds"] else ""
                    detail = f"{rec['keyword_output_class']} {rec['keyword_explanation']}".strip()
                    print(f"{rec['keyword_result']:<6} {secs:>7}  {detail}")
                    if args.review:
                        input("    Look at the browser, then press Enter to continue...")
                run_times.append({"llm": llm, "run_index": run_index, "seconds": round(time.time() - run_start, 1)})
        finally:
            driver.quit()

    grade_and_report(results, run_times, args.runs, use_judge=not args.no_judge, model=args.model)


if __name__ == "__main__":
    main()
