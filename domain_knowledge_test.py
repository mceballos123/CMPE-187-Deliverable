#!/usr/bin/env python3
"""
CMPE 187 Deliverable #3 - Domain Knowledge automation.

Usage:
    python3 domain_knowledge_test.py --app gemini --test-ids TC-DK-03 TC-DK-08 --ready
    python3 domain_knowledge_test.py --app all

Claude is collected manually when authentication/human verification blocks
Selenium. Put its eight records in results/raw/claude.json for the same judge.
Targeted retries preserve other rows, archive originals, and log failed attempts.

Output:
    results/raw/<app>.json
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from test_data import DOMAIN_TESTS
from domain_capture import capture_error

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

APP_CONFIG = {
    "chatgpt": {
        "url": "https://chatgpt.com/",
        "input": [
            (By.CSS_SELECTOR, "#prompt-textarea"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
            (By.CSS_SELECTOR, "textarea"),
        ],
        "send": [
            (By.CSS_SELECTOR, "button[data-testid='send-button']"),
            (By.CSS_SELECTOR, "button[aria-label*='Send']"),
            (By.CSS_SELECTOR, "button[aria-label*='send']"),
        ],
        "response": [
            (By.CSS_SELECTOR, "[data-message-author-role='assistant']"),
            (By.CSS_SELECTOR, "div[data-message-author-role='assistant']"),
            (By.CSS_SELECTOR, "article[data-turn='assistant']"),
            (By.CSS_SELECTOR, "article"),
            (By.CSS_SELECTOR, "div.markdown"),
            (By.CSS_SELECTOR, "div.prose"),
        ],
    },
    "claude": {
        "url": "https://claude.ai/new",
        "input": [
            (By.CSS_SELECTOR, "div.ProseMirror[contenteditable='true']"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
            (By.CSS_SELECTOR, "textarea"),
        ],
        "send": [
            (By.CSS_SELECTOR, "button[aria-label*='Send']"),
            (By.CSS_SELECTOR, "button[aria-label*='send']"),
        ],
        "response": [
            (By.CSS_SELECTOR, "div.font-claude-message"),
            (By.CSS_SELECTOR, "div[data-testid*='message']"),
            (By.CSS_SELECTOR, "article"),
        ],
    },
    "gemini": {
        "url": "https://gemini.google.com/app",
        "input": [
            (By.CSS_SELECTOR, "rich-textarea div[contenteditable='true']"),
            (By.CSS_SELECTOR, "div.ql-editor[contenteditable='true']"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
            (By.CSS_SELECTOR, "textarea"),
        ],
        "send": [
            (By.CSS_SELECTOR, "button.send-button"),
            (By.CSS_SELECTOR, "button[aria-label*='Send']"),
            (By.CSS_SELECTOR, "button[aria-label*='send']"),
        ],
        "response": [
            (By.CSS_SELECTOR, "model-response message-content"),
            (By.CSS_SELECTOR, "model-response .markdown-main-panel"),
            (By.CSS_SELECTOR, "model-response .response-content"),
        ],
        "assistant_only": True,
    },
    "perplexity": {
        "url": "https://www.perplexity.ai/",
        "input": [
            (By.CSS_SELECTOR, "textarea"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
        ],
        "send": [
            (By.CSS_SELECTOR, "button[aria-label*='Submit']"),
            (By.CSS_SELECTOR, "button[aria-label*='Send']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
        ],
        "response": [
            (By.CSS_SELECTOR, "div.prose"),
            (By.CSS_SELECTOR, "article"),
        ],
    },
}


def build_driver():
    options = webdriver.ChromeOptions()

    profile_dir = ROOT / "chrome_profile"
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument("--profile-directory=Default")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")

    return webdriver.Chrome(options=options)


def find_first(driver, selectors, timeout=25):
    end = time.time() + timeout
    while time.time() < end:
        for by, value in selectors:
            try:
                els = [e for e in driver.find_elements(by, value) if e.is_displayed()]
                if els:
                    return els[-1]
            except WebDriverException:
                pass
        time.sleep(0.5)
    raise TimeoutException("Could not find a visible element.")


def visible_texts(driver, selectors):
    out = []
    for by, value in selectors:
        try:
            for e in driver.find_elements(by, value):
                if e.is_displayed():
                    txt = e.text.strip()
                    if txt:
                        out.append(txt)
        except WebDriverException:
            pass
    return out


def body_text(driver):
    try:
        return driver.find_element(By.TAG_NAME, "body").text.strip()
    except Exception:
        return ""


def enter_prompt(el, prompt):
    el.click()
    modifier = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL
    try:
        el.send_keys(modifier, "a")
        el.send_keys(Keys.BACKSPACE)
    except Exception:
        pass
    el.send_keys(prompt)


def submit(driver, cfg):
    try:
        find_first(driver, cfg["send"], timeout=5).click()
    except TimeoutException:
        driver.switch_to.active_element.send_keys(Keys.ENTER)


def wait_response(driver, cfg, before_selector_texts, before_body, prompt, timeout=150):
    """
    1) Prefer known assistant-message selectors.
    2) Gemini never uses page-body fallback: it can capture sign-in chrome.
    """
    end = time.time() + timeout
    last = ""
    stable_since = None

    while time.time() < end:
        current = visible_texts(driver, cfg["response"])
        new_items = [t for t in current if t not in before_selector_texts and t.strip() != prompt.strip()]
        candidate = new_items[-1] if new_items else ""

        # DOM-selector fallback: compare page body before vs. after.
        if not candidate and not cfg.get("assistant_only"):
            now_body = body_text(driver)
            if now_body and now_body != before_body:
                # Grab text after the last occurrence of the submitted prompt.
                idx = now_body.rfind(prompt)
                if idx >= 0:
                    tail = now_body[idx + len(prompt):].strip()
                    # Remove obvious input-placeholder/footer text if present.
                    if len(tail) > 30:
                        candidate = tail

        if candidate and capture_error(candidate):
            candidate = ""
        if not candidate:
            last = ""
            stable_since = None
        elif candidate != last:
            last = candidate
            stable_since = time.time()
        elif stable_since and time.time() - stable_since >= 6:
            streaming = driver.find_elements(By.CSS_SELECTOR, "button[aria-label*='Stop'], button[aria-label*='stop'], .stop-icon")
            if not any(element.is_displayed() for element in streaming):
                return candidate

        time.sleep(1)

    raise TimeoutException("No stable assistant response captured; page/login text is not an answer.")


def run_app(driver, app, test_ids=None, ready=False):
    cfg = APP_CONFIG[app]
    driver.get(cfg["url"])

    print(f"\n[{app.upper()}] Browser opened.")
    if not ready:
        input("Log in if needed, open a fresh chat, then press ENTER here to continue... ")

    output_path = RAW_DIR / f"{app}.json"
    records = json.loads(output_path.read_text()) if output_path.exists() else []
    tests = [test for test in DOMAIN_TESTS if not test_ids or test["id"] in test_ids]
    if records:
        archive = RAW_DIR / "archive"
        archive.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (archive / f"{app}.{stamp}.json").write_bytes(output_path.read_bytes())
    attempts_path = ROOT / "results" / "domain_knowledge_capture_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    failures = 0

    for index, test in enumerate(tests, start=1):
        print(f"{test['id']} - {test['prompt']}")

        if index > 1:
            driver.get(cfg["url"])
            time.sleep(3)

        attempt = {"app": app, "test_id": test["id"],
                   "attempted_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            box = find_first(driver, cfg["input"], timeout=30)
            before_selectors = visible_texts(driver, cfg["response"])
            before_page = body_text(driver)
            enter_prompt(box, test["prompt"])
            submit(driver, cfg)
            response = wait_response(driver, cfg, before_selectors, before_page,
                                     test["prompt"], timeout=150)
            record = {
                "app": app,
                "test_id": test["id"],
                "prompt": test["prompt"],
                "expected": test["expected"],
                "response": response,
            }
            records = [row for row in records if row.get("test_id", row.get("id")) != test["id"]]
            records.append(record)
            records.sort(key=lambda row: row.get("test_id", row.get("id")))
            # Save after every successful capture so a later error cannot lose it.
            output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
            attempt.update(status="CAPTURED", response=response)
            print(f"  captured {len(response)} characters", flush=True)
        except WebDriverException as exc:
            failures += 1
            attempt.update(status="AUTOMATION_ERROR", error=f"{type(exc).__name__}: {exc.msg}")
            print(f"  AUTOMATION_ERROR: {exc.msg}", flush=True)
        attempts.append(attempt)
        attempts_path.write_text(json.dumps(attempts, indent=2, ensure_ascii=False) + "\n")
        time.sleep(2)

    print(f"\nRaw results: {output_path} | failed captures: {failures}", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app",
        choices=["chatgpt", "claude", "gemini", "perplexity", "all"],
        default="all",
    )
    parser.add_argument("--test-ids", nargs="+", choices=[test["id"] for test in DOMAIN_TESTS])
    parser.add_argument("--ready", action="store_true", help="Use the existing login session without an interactive pause")
    args = parser.parse_args()

    if args.app == "claude":
        parser.exit(0, "Claude is a manual collection exception: save all eight records in results/raw/claude.json, then run evaluate_domain_results.py.\n")
    if args.app == "all":
        print("Claude: manual collection exception; its raw file is evaluated separately.", flush=True)
    # Do not retry blocked Claude authentication during an all-app run.
    apps = [app for app in APP_CONFIG if app != "claude"] if args.app == "all" else [args.app]
    driver = build_driver()
    failures = 0

    try:
        for app in apps:
            try:
                failures += run_app(driver, app, args.test_ids, args.ready)
            except Exception as exc:
                failures += 1
                print(f"\nERROR for {app}: {exc}")
                print("The site UI may have changed; update that app's selectors.")
    finally:
        driver.quit()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
