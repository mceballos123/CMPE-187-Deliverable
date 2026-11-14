#!/usr/bin/env python3
"""
Selenium test script: memory / context retention on ChatGPT, Claude, Gemini, Perplexity.

Drives the real web UIs (no APIs). No login step: it just opens each page. If the chat
box isn't available (e.g. login wall), that test is recorded as ERROR.

Usage:
    pip install selenium
    python llm_memory_test.py                         # all LLMs, all tests
    python llm_memory_test.py --llms claude gemini    # subset of LLMs
    python llm_memory_test.py --tests IM-01 IM-10     # subset of tests
    python llm_memory_test.py --timeout 240

Output (./results/run_<timestamp>/): results.json, results.csv, screenshots/*.png

NOTE: Site selectors change often. If a site breaks, edit the SITES dict below.
NOTE: Pass/fail is a keyword/regex heuristic mirroring the pass criteria. Review the
      saved responses/screenshots before reporting - LLM answers vary.
"""
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import (
    ElementNotInteractableException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_DIR = Path(__file__).resolve().parent
NEXT_FILE = BASE_DIR / "next"  # created by you (or Claude) to continue in --review mode

# --------------------------------------------------------------------------- #
# Site configuration (CSS selectors; each is a list tried in order)
# --------------------------------------------------------------------------- #
SITES = {
    "ChatGPT": {
        "url": "https://chatgpt.com/",
        "input": ["#prompt-textarea", "div#prompt-textarea[contenteditable='true']", "textarea"],
        "response": ["li[data-message-role='assistant']", "div[data-message-author-role='assistant']"],
        "busy": ["button[data-testid='stop-button']", "button[aria-label*='Stop']"],
    },
    "Claude": {
        "url": "https://claude.ai/new",
        "manual_login": True,  # wait (up to 10 min) for you to pass the check / log in by hand
        "input": ["div[contenteditable='true'].ProseMirror", "div[contenteditable='true']"],
        "response": ["div.font-claude-response", "div[data-testid='assistant-message']",
                     "div.font-claude-message"],
        "busy": ["button[aria-label='Stop response']", "[data-is-streaming='true']"],
    },
    "Gemini": {
        "url": "https://gemini.google.com/app",
        "input": ["div.ql-editor[contenteditable='true']", "rich-textarea div[contenteditable='true']",
                  "div[contenteditable='true']"],
        "response": ["message-content", "div.model-response-text", "model-response"],
        "busy": ["button[aria-label='Stop response']", "button.stop", "mat-icon[fonticon='stop']"],
    },
    "Perplexity": {
        "url": "https://www.perplexity.ai/",
        "blocker_text": "sign in",  # logged out (sidebar says "Sign In"): wait for you to sign in by hand
        "input": ["div#ask-input[contenteditable='true']", "textarea[placeholder]",
                  "div[contenteditable='true'][role='textbox']", "div[contenteditable='true']"],
        "response": ["div[id^='markdown-content']", "div.prose"],
        "busy": ["button[aria-label='Stop']", "button[aria-label='Stop generating']"],
    },
}

# --------------------------------------------------------------------------- #
# Test cases. "prompts" = ordered list; the LAST one is the follow-up under test.
# --------------------------------------------------------------------------- #
TESTS = [
    {
        "id": "IM-01",
        "prompts": [
            "What is the purpose of an SNMP trap, and how is it different from SNMP polling?",
            "Based on the previous explanation of SNMP traps, what is the reason for traps having a different port number than polling?",
        ],
        "expected": "The output should recall the previous explanation of SNMP traps and identify why the port numbers differ.",
        "criteria": "The response correctly identifies UDP port 161 and UDP port 162.",
    },
    {
        "id": "IM-02",
        "prompts": [
            "For the rest of the conversation, explain topics related to SNMP in simple but technical terms.",
            "What is the purpose of SNMP?",
        ],
        "expected": "The response should follow the earlier instruction and provide a simple explanation of SNMP.",
        "criteria": "The response explains SNMP in simple language while remaining technically accurate.",
    },
    {
        "id": "IM-03",
        "prompts": [
            "Explain SNMP and the five functional areas of FCAPS.",
            "During our conversation, I mentioned SNMP and FCAPS. For my exam tomorrow, please provide the most important information from our conversation as a review.",
        ],
        "expected": "The output should remember the previous discussion about SNMP and FCAPS and provide a relevant review of the important information mentioned.",
        "criteria": "The response recalls and refreshes important concepts about SNMP and FCAPS.",
    },
    {
        "id": "IM-04",
        "prompts": [
            "Explain the five functional areas of FCAPS and what each area is responsible for.",
            "Based on the previous explanation of FCAPS, what does the 'P' in FCAPS represent?",
        ],
        "expected": "The response should remember the previous explanation of FCAPS and identify the meaning of the letter \"P\".",
        "criteria": "The response correctly identifies \"P\" as Performance Management and explains its purpose in network management.",
    },
    {
        "id": "IM-05",
        "prompts": [
            "What is the difference between SNMP polling and SNMP traps?",
            "Earlier in our conversation, we discussed the difference between SNMP polling and SNMP traps. Which method sends an unsolicited notification when a network event occurs?",
        ],
        "expected": "The response should use the previous conversation context to identify the method that sends an unsolicited notification.",
        "criteria": "The response correctly identifies SNMP traps and explains that they send notifications without waiting for a polling request.",
    },
    {
        "id": "IM-09",
        "prompts": [
            "What is the difference between SNMP polling and SNMP traps?",
            "Which UDP ports do they use?",
            "If a link on one of my switches goes down, which of the two would let the NMS know first, and why?",
        ],
        "expected": "The response should understand that \"they\" and \"the two\" refer to polling and traps from earlier in the conversation, without the user naming them again.",
        "criteria": "The response identifies SNMP traps and explains that the switch sends a trap as soon as the event happens, while polling has to wait for the next poll. The test fails if the response asks which two methods the user means.",
    },
    {
        "id": "IM-10",
        "prompts": [
            "I'm setting up SNMP on our core switch, SW-CORE-01. Its management IP is 10.20.0.5 and our NMS is at 10.20.0.50.",
            "What settings do I need to configure on the switch so it can send SNMPv2c traps to the NMS?",
            "What community string did I say we use for SW-CORE-01?",
        ],
        "expected": "The response should recognize that a community string was never given in the conversation and say so, instead of making one up.",
        "criteria": "The response states that no community string was provided. It may suggest a common default or ask the user for theirs, as long as it does not present a value as something the user said. The test fails if the response claims the user gave a community string.",
    },
]

# The SAME three unrelated "distractor" prompts are used for every test, inserted just before
# the follow-up when --fillers 3. They are general networking topics but deliberately avoid
# anything the IM tests touch (SNMP, FCAPS, traps/polling, ports, community strings), and are
# single-topic so they can't confuse pronouns like "the two" in the follow-ups.
FILLERS = [
    "What are the 7 layers of the OSI model?",
    "What is a VLAN and why would I use one?",
    "What does a router do?",
]

# --------------------------------------------------------------------------- #
# Pass/fail evaluators: return (passed: bool, explanation: str)
# --------------------------------------------------------------------------- #
def _has(text, pattern):
    return re.search(pattern, text, re.I) is not None


def _is_clarification(text):
    return _has(text, r"which (two|methods|ones)|what (two|methods) (do|are) you|could you (clarify|specify)|"
                      r"can you (clarify|specify)|do you mean|not sure what (you|they)|please clarify")


def eval_im01(r):
    miss = [p for p in ("161", "162") if p not in r]
    if miss:
        return False, f"Response does not mention UDP port(s): {', '.join(miss)}."
    if not _has(r, r"\budp\b"):
        return False, "Ports 161/162 mentioned but UDP not identified."
    return True, ""


def eval_im02(r):
    problems = []
    if not _has(r, r"\bsnmp\b"):
        problems.append("does not mention SNMP")
    if not _has(r, r"monitor|manag|collect|query|configur"):
        problems.append("does not state the monitoring/management purpose")
    if not _has(r, r"device|network|router|switch|agent|nms|manager"):
        problems.append("lacks technical network terms")
    if len(r.split()) > 450:
        problems.append("answer is long (>450 words), may not be 'simple'")
    return (not problems, "; ".join(problems).capitalize() + "." if problems else "")


def eval_im03(r):
    areas = {"fault": r"\bfault\b", "configuration": r"\bconfiguration\b", "accounting": r"\baccounting\b|\bbilling\b",
             "performance": r"\bperformance\b", "security": r"\bsecurity\b"}
    missing = [k for k, p in areas.items() if not _has(r, p)]
    problems = []
    if not _has(r, r"\bsnmp\b"):
        problems.append("SNMP not mentioned")
    if not _has(r, r"\bfcaps\b"):
        problems.append("FCAPS not mentioned")
    if len(missing) > 1:
        problems.append(f"missing FCAPS areas: {', '.join(missing)}")
    return (not problems, "; ".join(problems).capitalize() + "." if problems else "")


def eval_im04(r):
    if not _has(r, r"performance"):
        return False, "Response does not identify 'P' as Performance."
    if not _has(r, r"monitor|measure|throughput|latency|utiliz|bandwidth|metric|optimi|capacity|response time"):
        return False, "Performance named but its network-management purpose is not explained."
    return True, ""


def eval_im05(r):
    if not _has(r, r"\btraps?\b"):
        return False, "Response does not identify SNMP traps."
    if not _has(r, r"unsolicited|without (waiting|being (polled|asked|requested))|asynchronous|on its own|"
                   r"push|initiat|not wait|no (poll|request)|without a (poll|request)|without.{0,25}request|"
                   r"agent[- ]initiated|sent immediately|on their own"):
        return False, "Traps identified but not explained as sent without waiting for a polling request."
    return True, ""


def eval_im09(r):
    if _is_clarification(r):
        return False, "Response asks the user to clarify which two methods are meant."
    if not _has(r, r"\btraps?\b"):
        return False, "Response does not identify SNMP traps."
    if not _has(r, r"as soon as|immediate|instant|right away|real[- ]time|at once|the moment|when the (event|link)|"
                   r"without waiting|asynchronous|unsolicited"):
        return False, "Traps identified but no explanation that the trap is sent as soon as the event happens."
    if not _has(r, r"next poll|poll(ing)? (interval|cycle)|until (the|it)|wait|delay|periodic|interval"):
        return False, "Does not explain that polling must wait for the next poll."
    return True, ""


def eval_im10(r):
    claim_re = re.compile(
        r"\byou\s+(?:said|mentioned|told|stated|provided|specified|gave|indicated|noted|use|are using|use[d]?)\b.{0,80}"
        r"(?:community|public|private)", re.I)
    neg_re = re.compile(r"\b(?:didn'?t|did not|haven'?t|have not|never|not|no|wasn'?t|weren'?t|don'?t|do not|"
                        r"unable|cannot|can'?t)\b", re.I)
    for s in re.split(r"(?<=[.!?\n])\s+", r):
        if claim_re.search(s) and not neg_re.search(s):
            return False, f"Response claims the user gave a community string: \"{s.strip()[:160]}\""
    denies = _has(r, r"(didn'?t|did not|haven'?t|have not|never|not)\s+(?:\w+\s+){0,4}"
                     r"(mention|provide|specif|share|say|said|tell|told|give|gave|state|includ)|"
                     r"no community string|not (been )?(provided|specified|mentioned|given|shared)|"
                     r"(don'?t|do not) (see|have|know)|wasn'?t (provided|given|mentioned|specified)")
    if not denies:
        return False, "Response does not state that no community string was provided."
    return True, ""


EVALUATORS = {"IM-01": eval_im01, "IM-02": eval_im02, "IM-03": eval_im03, "IM-04": eval_im04,
              "IM-05": eval_im05, "IM-09": eval_im09, "IM-10": eval_im10}

# --------------------------------------------------------------------------- #
# Selenium helpers
# --------------------------------------------------------------------------- #
def make_driver(llm, keep_open=False):
    opts = webdriver.ChromeOptions()
    profile = BASE_DIR / "chrome_profiles" / llm.lower()
    profile.mkdir(parents=True, exist_ok=True)
    opts.add_argument(f"--user-data-dir={profile}")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    if keep_open:
        opts.add_experimental_option("detach", True)  # leave Chrome open after the script ends
    return webdriver.Chrome(options=opts)  # Selenium Manager fetches chromedriver


def find_first(driver, selectors, visible=True):
    """Return first (displayed) element matching any selector, else None."""
    for sel in selectors:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, sel):
                if not visible or el.is_displayed():
                    return el
        except WebDriverException:
            continue
    return None


def find_all_responses(driver, selectors):
    for sel in selectors:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            return els
    return []


def is_busy(driver, cfg):
    return find_first(driver, cfg["busy"]) is not None


def safe_text(el):
    try:
        return el.text.strip()
    except StaleElementReferenceException:
        return ""


def wait_for_input(driver, cfg, timeout=30):
    return WebDriverWait(driver, timeout, poll_frequency=0.5).until(
        lambda d: find_first(d, cfg["input"]) or False)


def _page_has(driver, text):
    try:
        return text.lower() in driver.find_element(By.TAG_NAME, "body").text.lower()
    except WebDriverException:
        return False


def dismiss_cookie_banner(driver):
    for el in driver.find_elements(By.XPATH, "//button[normalize-space()='Only necessary']"):
        try:
            el.click()
            return
        except WebDriverException:
            continue


def open_new_chat(driver, cfg):
    driver.get(cfg["url"])
    WebDriverWait(driver, 15, poll_frequency=0.5).until(lambda d: find_first(d, cfg["input"]) or _page_has(d, "log in"))
    dismiss_cookie_banner(driver)
    blocker = cfg.get("blocker_text")
    if blocker and _page_has(driver, blocker):
        print("\n    >>> Not signed in. Sign in in the browser window (try 'Continue with email' if Google "
              "is refused); the test continues automatically.", flush=True)
        WebDriverWait(driver, 600, poll_frequency=1).until(lambda d: not _page_has(d, blocker))
        time.sleep(2)  # let the sign-in finish saving before reloading
        driver.get(cfg["url"])
        wait_for_input(driver, cfg, timeout=30)
    if cfg.get("manual_login"):
        if find_first(driver, cfg["input"]) is None:
            print("\n    >>> Pass the human check / log in in the browser window. "
                  "The test continues automatically once the chat box appears.", flush=True)
        wait_for_input(driver, cfg, timeout=600)
    else:
        wait_for_input(driver, cfg, timeout=30)


def send_prompt(driver, cfg, text):
    box = wait_for_input(driver, cfg)
    box.click()
    box.send_keys(text)
    time.sleep(0.3)  # let the editor register the text before Enter
    box.send_keys(Keys.ENTER)


def wait_for_response(driver, cfg, prev_count, timeout, stable_secs=3.0, prev_text=""):
    """Wait for a new response block, then for generation to stop and text to settle.
    A new answer = more blocks than before, OR a different last block (some sites, e.g.
    Perplexity, unload old answers in long chats so the block count can stay flat)."""
    def new_block(d):
        els = find_all_responses(d, cfg["response"])
        if len(els) > prev_count:
            return True
        return bool(els) and safe_text(els[-1]) not in ("", prev_text)
    WebDriverWait(driver, 60, poll_frequency=0.5).until(new_block)

    deadline = time.time() + timeout
    last_text, last_change = "", time.time()
    while time.time() < deadline:
        els = find_all_responses(driver, cfg["response"])
        text = safe_text(els[-1]) if els else ""
        if text != last_text:
            last_text, last_change = text, time.time()
        if text and not is_busy(driver, cfg) and time.time() - last_change >= stable_secs:
            return text
        time.sleep(0.5)
    raise TimeoutException(f"Response did not finish within {timeout}s")


def screenshot_final(driver, cfg, path):
    els = find_all_responses(driver, cfg["response"])
    if els:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'end'});", els[-1])
            WebDriverWait(driver, 3, poll_frequency=0.25).until(lambda d: True)
        except WebDriverException:
            pass
    driver.save_screenshot(str(path))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_test(driver, llm, cfg, test, timeout, shots_dir, fillers=0):
    ts = datetime.now().isoformat(timespec="seconds")
    shot_name = f"{llm}_{test['id']}_f{fillers}_{datetime.now():%H%M%S}.png"
    # filler prompts go between the setup prompt(s) and the follow-up under test
    prompts = test["prompts"][:-1] + FILLERS[:fillers] + test["prompts"][-1:]
    record = {
        "llm": llm, "test_id": test["id"], "filler_count": fillers, "collection_method": "automated",
        "filler_prompts": json.dumps(FILLERS[:fillers], ensure_ascii=False),
        "initial_prompt": test["prompts"][0],
        "second_prompt": test["prompts"][1] if len(test["prompts"]) > 2 else "",
        "follow_up_prompt": test["prompts"][-1],
        "expected_result": test["expected"], "pass_criteria": test["criteria"],
        "conversation_context": "", "actual_response": "",
        "result": "FAIL", "failure_explanation": "", "timestamp": ts, "screenshot": shot_name,
    }
    context = []
    try:
        open_new_chat(driver, cfg)
        responses = []
        for prompt in prompts:
            before = find_all_responses(driver, cfg["response"])
            prev, prev_text = len(before), (safe_text(before[-1]) if before else "")
            send_prompt(driver, cfg, prompt)
            reply = wait_for_response(driver, cfg, prev, timeout, prev_text=prev_text)
            context.append({"role": "user", "text": prompt})
            context.append({"role": "assistant", "text": reply})
            responses.append(reply)
        final = responses[-1]
        record["actual_response"] = final
        record["conversation_context"] = json.dumps(context[:-1], ensure_ascii=False)
        passed, why = EVALUATORS[test["id"]](final)
        record["result"] = "PASS" if passed else "FAIL"
        record["failure_explanation"] = why
    except Exception as e:  # keep going on errors; record them
        record["result"] = "ERROR"
        record["failure_explanation"] = f"{type(e).__name__}: {e}"
        record["conversation_context"] = json.dumps(context, ensure_ascii=False)
    finally:
        try:
            screenshot_final(driver, cfg, shots_dir / shot_name)
        except WebDriverException as e:
            record["screenshot"] = f"(screenshot failed: {e})"
    return record


FIELDS = ["llm", "test_id", "collection_method", "filler_count", "filler_prompts", "initial_prompt", "second_prompt", "follow_up_prompt", "expected_result",
          "pass_criteria", "conversation_context", "actual_response", "result", "failure_explanation",
          "timestamp", "screenshot"]


def save(results, out_dir):
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llms", nargs="+", default=list(SITES), choices=list(SITES), metavar="LLM",
                    help=f"Subset of: {', '.join(SITES)}")
    ap.add_argument("--tests", nargs="+", default=[t["id"] for t in TESTS], help="Test IDs, e.g. IM-01 IM-10")
    ap.add_argument("--fillers", nargs="+", type=int, default=[0, 3], choices=range(len(FILLERS) + 1),
                    metavar="N", help=f"Unrelated prompts to insert before the follow-up (0-{len(FILLERS)}). "
                                      "Several values run each test once per value. Default: 0 3")
    ap.add_argument("--review", action="store_true",
                    help=f"After each run, keep the browser open until the file {NEXT_FILE.name} is created")
    ap.add_argument("--keep-open", action="store_true", help="Leave the browser window open when finished")
    ap.add_argument("--timeout", type=int, default=180, help="Max seconds to wait per response")
    args = ap.parse_args()

    unknown = set(args.tests) - {t["id"] for t in TESTS}
    if unknown:
        sys.exit(f"Unknown test IDs: {', '.join(sorted(unknown))}")
    tests = [t for t in TESTS if t["id"] in args.tests]

    out_dir = BASE_DIR / "results" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    shots_dir = out_dir / "screenshots"
    shots_dir.mkdir(parents=True)
    results = []

    for llm in args.llms:
        cfg = SITES[llm]
        print(f"\n=== {llm} ===")
        driver = make_driver(llm, keep_open=args.keep_open)
        try:
            for test in tests:
                for n in args.fillers:
                    print(f"  {test['id']} (fillers={n}) ...", end=" ", flush=True)
                    rec = run_test(driver, llm, cfg, test, args.timeout, shots_dir, fillers=n)
                    results.append(rec)
                    save(results, out_dir)  # save after every run so nothing is lost
                    print(rec["result"], f"- {rec['failure_explanation']}" if rec["failure_explanation"] else "")
                    if args.review:  # keep the browser open until the NEXT_FILE appears
                        print(f"    Review the browser, then: touch {NEXT_FILE}", flush=True)
                        while not NEXT_FILE.exists():
                            time.sleep(1)
                        NEXT_FILE.unlink()
        finally:
            if not args.keep_open:
                driver.quit()

    passed = sum(r["result"] == "PASS" for r in results)
    print(f"\nDone: {passed}/{len(results)} passed. Results in {out_dir}")


if __name__ == "__main__":
    main()
