#!/usr/bin/env python3
"""Selenium test execution for the Question/Answering 3D test suite.

One-time setup (a Chrome window opens per app, sign in, press Enter; the profile is kept):
  python qa_runner.py --login

Check that every site's page elements are found and an answer can be read:
  python qa_runner.py --probe

Runs:
  python qa_runner.py --plan core --apps chatgpt --repeat 1     # short demo
  python qa_runner.py --plan 3d --repeat 3                      # Deliverable 3 run, all 4 apps
  python qa_runner.py --resume results/run_20261123_181500      # finish an interrupted run

Each execution: open a fresh chat, send the context filler turns (QA-CX-02 only), send the
test prompt, wait until the answer stops streaming, save the text and a screenshot, then
grade it with the oracle (and the Ollama judge if it is running). Results go to
results/<run_id>/results.jsonl, one line per execution.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import qa_generate
import qa_judge
import qa_sites

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
PROFILES = BASE / "chrome_profiles"
SCRIPT_FILES = ["qa_generate.py", "qa_oracle.py", "qa_judge.py", "qa_sites.py", "qa_runner.py",
                "data/qa_model_3d.json", "data/qa_test_data.json"]

PRINT_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()


ALL_APPS = ["chatgpt", "claude", "gemini", "perplexity"]

def log(app, msg):
    with PRINT_LOCK:
        print("[%s %-10s] %s" % (datetime.now().strftime("%H:%M:%S"), app, msg), flush=True)


# ------------------------------------------------------------------ browser

def make_driver(app, args):
    from selenium import webdriver
    opts = webdriver.ChromeOptions()
    profiles = Path(args.profiles_dir) if args.profiles_dir else PROFILES
    profiles.mkdir(exist_ok=True)
    opts.add_argument("--user-data-dir=%s" % (profiles / app).resolve())
    opts.add_argument("--profile-directory=Default")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    if args.headless:
        opts.add_argument("--headless=new")
    for extra in args.chrome_arg or []:
        opts.add_argument(extra)
    if args.chrome_binary:
        opts.binary_location = args.chrome_binary
    service = None
    if args.chromedriver:
        from selenium.webdriver.chrome.service import Service
        service = Service(args.chromedriver)
    try:
        driver = webdriver.Chrome(options=opts, service=service)
    except Exception as e:
        raise RuntimeError("Chrome did not start for %s. If another run or a Chrome window is using "
                           "%s, close it and try again. (%s)" % (app, profiles / app, str(e).splitlines()[0])) from e
    driver.set_page_load_timeout(60)
    return driver


def find_first(driver, selectors, visible=True):
    from selenium.webdriver.common.by import By
    for sel in selectors:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, sel):
                if not visible or el.is_displayed():
                    return el, sel
        except Exception:
            continue
    return None, None


def find_all(driver, selectors):
    """Elements for the first selector that matches anything."""
    from selenium.webdriver.common.by import By
    for sel in selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            continue
        if els:
            return els, sel
    return [], None


def body_text(driver):
    from selenium.webdriver.common.by import By
    try:
        return driver.find_element(By.TAG_NAME, "body").text
    except Exception:
        return ""


def blocked_reason(driver):
    try:
        title = driver.title or ""
    except Exception:
        title = ""
    text = (title + "\n" + body_text(driver)).lower()
    for reason in ("challenge", "limit", "login"):
        if re.search(qa_sites.BLOCK_PATTERNS[reason], text):
            return reason
    return None


def wait_for_input(driver, site, args, app, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        el, _sel = find_first(driver, site["input"])
        if el is not None:
            return el
        time.sleep(1)
    reason = blocked_reason(driver) or "no input box found"
    if args.pause and reason != "limit":
        # same idea as Martin's script: wait (up to 10 min) for the tester to sign in or pass
        # the check by hand in that window; the run goes on by itself once the chat box shows
        log(app, ">>> Blocked (%s). Sign in or pass the check yourself in the %s window. "
                 "The run continues when the chat box appears." % (reason, site["name"]))
        end = time.time() + 600
        while time.time() < end:
            el, _sel = find_first(driver, site["input"])
            if el is not None:
                time.sleep(2)
                return el
            time.sleep(2)
    raise BlockedError(reason)


# Optional pop-ups that block the chat box: cookie banners (the most private choice) and
# "log in for more features" nudges on logged-out sites. Never a sign-in or a human check.
DISMISS_LABELS = ("Only necessary", "Reject non-essential cookies", "Reject all", "Not now",
                  "Stay logged out", "No thanks", "Maybe later")


def dismiss_popups(driver):
    """Click the dismiss button of a visible optional pop-up. Returns True if one was clicked."""
    from selenium.webdriver.common.by import By
    for label in DISMISS_LABELS:
        for el in driver.find_elements(By.XPATH, "//button[normalize-space()='%s']" % label):
            try:
                if el.is_displayed():
                    el.click()
                    time.sleep(1)
                    return True
            except Exception:
                continue
    return False


class BlockedError(Exception):
    pass


def type_prompt(driver, el, text):
    from selenium.webdriver.common.keys import Keys
    try:
        el.click()
    except Exception:  # covered by another element (Perplexity's follow-up bar): focus it directly
        driver.execute_script("arguments[0].scrollIntoView({block:'center'}); arguments[0].focus();", el)
    try:
        el.send_keys(text)
    except Exception:
        pass
    time.sleep(0.5)
    current = (el.get_attribute("value") or el.text or "")
    if text[:30] not in current:  # some editors ignore send_keys; insert the text like a paste
        driver.execute_script("arguments[0].focus(); document.execCommand('insertText', false, arguments[1]);",
                              el, text)
        time.sleep(0.5)
    try:
        el.send_keys(Keys.ENTER)
    except Exception:
        driver.switch_to.active_element.send_keys(Keys.ENTER)


def safe_text(el):
    try:
        return el.text.strip()
    except Exception:  # element re-rendered while streaming
        return ""


def clean_body_answer(body, prompt):
    """Fallback extraction: the page text after the last copy of the prompt."""
    for probe in (prompt, prompt[:60]):
        idx = body.rfind(probe)
        if idx >= 0:
            rest = body[idx + len(probe):]
            break
    else:
        return ""
    lines = [ln for ln in rest.split("\n") if ln.strip().lower() not in qa_sites.BOILERPLATE]
    return "\n".join(lines).strip()


def send_and_wait(driver, site, app, text, args):
    """Send one message and return (answer, extraction, secs, timed_out)."""
    dismiss_popups(driver)
    el = wait_for_input(driver, site, args, app)
    before, _ = find_all(driver, site["response"])
    n_before = len(before)
    prev_text = safe_text(before[-1]) if before else ""
    type_prompt(driver, el, text)
    t0 = time.time()
    last, last_change, extraction = "", time.time(), "selector"
    while True:
        time.sleep(1)
        els, _sel = find_all(driver, site["response"])
        cur = safe_text(els[-1]) if els else ""
        # a new answer = more answer blocks than before, or a different last block
        # (some sites, e.g. Perplexity, unload old answers so the count stays flat)
        if len(els) > n_before or (cur and cur != prev_text):
            txt = cur
            extraction = "selector"
        elif time.time() - t0 > 25:
            txt = clean_body_answer(body_text(driver), text)
            extraction = "body-diff"
        else:
            txt = ""
            if time.time() - t0 > 6 and dismiss_popups(driver):
                box, _sel = find_first(driver, site["input"])
                if box is not None and text[:30] in (box.get_attribute("value") or box.text or ""):
                    from selenium.webdriver.common.keys import Keys
                    box.send_keys(Keys.ENTER)  # the pop-up swallowed the first Enter
        if txt != last:
            last, last_change = txt, time.time()
        busy, _ = find_first(driver, site["busy"])
        if last and busy is None and time.time() - last_change >= args.stable:
            return last, extraction, round(last_change - t0, 1), False
        if time.time() - t0 > args.timeout:
            return last, extraction, round(time.time() - t0, 1), True


# --------------------------------------------------------------- execution

def execute(driver, site, app, test, repeat, run_dir, args):
    rec = {k: test[k] for k in ("test_id", "input_id", "base_id", "input_class", "context_class",
                                "augmentation", "rule", "requirements", "complexity", "expected_output",
                                "prompt")}
    rec.update(run_id=run_dir.name, app=app, app_name=site["name"], repeat=repeat,
               started_at=datetime.now().isoformat(timespec="seconds"), status="ok", error=None,
               turns=1 + len(test["fillers"]), test=test)
    t0 = time.time()
    try:
        driver.get(site["url"])
        time.sleep(2)
        dismiss_popups(driver)
        for filler in test["fillers"]:
            send_and_wait(driver, site, app, filler, args)
            time.sleep(args.delay)
        answer, extraction, secs, timed_out = send_and_wait(driver, site, app, test["prompt"], args)
        rec.update(answer=answer, extraction=extraction, response_secs=secs, url=driver.current_url)
        if timed_out:
            rec["status"] = "timeout"
            rec["error"] = "answer still changing after %ss" % args.timeout
        if not answer:
            reason = blocked_reason(driver)
            rec["status"] = "blocked" if reason else "error"
            rec["error"] = reason or "no answer text found"
        elif qa_sites.is_site_error(answer):
            rec["status"] = "error"
            rec["error"] = "the site returned an error message instead of an answer"
        elif extraction == "body-diff" and blocked_reason(driver):
            # no answer block appeared and the page shows a sign-in, limit or check prompt:
            # the text read is page chrome, not an answer, so it is not graded
            rec["status"] = "blocked"
            rec["error"] = "%s prompt on the page, no answer" % blocked_reason(driver)
    except BlockedError as e:
        rec.update(status="blocked", error=str(e), answer="")
    except Exception as e:
        rec.update(status="error", error="%s: %s" % (type(e).__name__, str(e)[:300]), answer=rec.get("answer", ""))
    rec["exec_secs"] = round(time.time() - t0, 1)
    shot = run_dir / "screens" / ("%s_%s_r%d.png" % (app, test["test_id"], repeat))
    try:
        els, _sel = find_all(driver, site["response"])
        if els:
            driver.execute_script("arguments[0].scrollIntoView({block:'end'});", els[-1])
            time.sleep(0.5)
        driver.save_screenshot(str(shot))
        rec["screenshot"] = str(shot.relative_to(run_dir))
    except Exception:
        rec["screenshot"] = None
    (run_dir / "answers" / ("%s_%s_r%d.txt" % (app, test["test_id"], repeat))).write_text(
        "PROMPT:\n%s\n\nANSWER:\n%s\n" % (test["prompt"], rec.get("answer", "")), encoding="utf-8")
    if rec["status"] in ("ok", "timeout") and rec.get("answer"):
        o, j, final, agreement = qa_judge.grade(rec["answer"], test, args.judge, args.judge_model, args.judge_url)
        rec.update(oracle=o, judge=j, final_verdict=final, agreement=agreement)
    else:
        rec.update(oracle=None, judge=None, final_verdict="NOT RUN", agreement=None)
    return rec


def done_keys(run_dir, redo_body_diff=False):
    """Executions whose latest record finished. With redo_body_diff, answers read by the
    body-diff fallback are run again too."""
    path = run_dir / "results.jsonl"
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["app"], r["test_id"], r["repeat"])] = r
    return {k for k, r in latest.items() if r.get("status") == "ok"
            and not qa_sites.is_site_error(r.get("answer"))
            and not (redo_body_diff and r.get("extraction") == "body-diff")}


def run_app(app, site, tests, run_dir, args, skip):
    todo = [(r, t) for r in range(1, args.repeat + 1) for t in tests if (app, t["test_id"], r) not in skip]
    if not todo:
        log(app, "nothing left to run")
        return
    log(app, "%d executions to run (%s)" % (len(todo), site["setup"]))
    driver = None
    try:
        driver = make_driver(app, args)
        for i, (repeat, test) in enumerate(todo, 1):
            rec = execute(driver, site, app, test, repeat, run_dir, args)
            if rec["status"] == "error":  # one retry: site hiccups are common
                log(app, "%s r%d: %s, retrying once" % (test["test_id"], repeat, rec["error"][:80]))
                time.sleep(args.delay * 3)
                rec = execute(driver, site, app, test, repeat, run_dir, args)
            with WRITE_LOCK, open(run_dir / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            o = rec.get("oracle") or {}
            log(app, "%3d/%d %-18s r%d %-8s %-12s %s %s" % (
                i, len(todo), test["test_id"], repeat, rec["status"], rec["final_verdict"],
                o.get("output_class", ""), ("(%s)" % rec["error"]) if rec["error"] else ""))
            if rec["status"] == "blocked" and rec["error"] == "limit":
                log(app, "usage limit reached, stopping this app. Use --resume later to finish.")
                break
            time.sleep(args.delay)
    except Exception:
        log(app, "stopped by an error:\n" + traceback.format_exc())
    finally:
        if driver:
            driver.quit()


def file_hashes():
    out = {}
    for name in SCRIPT_FILES:
        p = BASE / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return out


def load_suite(args):
    model, data = qa_generate.load("qa_model_3d.json"), qa_generate.load("qa_test_data.json")
    suite = qa_generate.build_suite(model, data, args.plan, args.seed)
    tests = suite["tests"]
    if args.filter:
        tests = [t for t in tests if re.search(args.filter, t["test_id"])]
    if args.max_tests:
        tests = tests[:args.max_tests]
    return suite, tests


def cmd_run(args):
    sites = qa_sites.load_sites(BASE)
    if args.resume:
        run_dir = Path(args.resume)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for k in ("plan", "seed", "repeat", "filter", "max_tests"):
            setattr(args, k, cfg["args"].get(k))
        if not args.apps:  # --apps given on resume runs only those apps
            args.apps = cfg["args"].get("apps")
        tests = cfg["tests"]
        skip = done_keys(run_dir, args.redo_body_diff)
        log("runner", "resuming %s, %d executions already done" % (run_dir.name, len(skip)))
    else:
        suite, tests = load_suite(args)
        run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = RESULTS / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            import selenium
            sel_version = selenium.__version__
        except ImportError:
            sel_version = None
        cfg = {
            "run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"),
            "args": {k: v for k, v in vars(args).items() if k not in ("func",)},
            "suite_meta": suite["meta"], "tests": tests,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "selenium": sel_version, "script_sha256": file_hashes()},
            "app_tiers": dict(kv.split("=", 1) for kv in (args.tier or [])),
            "sites": {a: {"url": sites[a]["url"]} for a in args.apps},
        }
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        skip = set()
    (run_dir / "screens").mkdir(exist_ok=True)
    (run_dir / "answers").mkdir(exist_ok=True)

    judge_state = qa_judge.judge("ok", {"prompt": "ping", "reference_key_points": ["ok"], "expected_output":
                                        "QA-OC-01"}, args.judge_model, args.judge_url, timeout=60) \
        if args.judge else {"status": "off"}
    if args.judge and judge_state["status"] != "ok":
        log("runner", "Ollama judge not available (%s). Grading with the oracle only."
            % judge_state.get("error", judge_state["status"]))
        args.judge = False
    n = len(tests) * len(args.apps) * args.repeat
    log("runner", "plan %s: %d tests x %d apps x %d repeats = %d executions, judge %s, run dir %s" % (
        args.plan, len(tests), len(args.apps), args.repeat, n, "on" if args.judge else "off", run_dir))

    t0 = time.time()
    threads = []
    for app in args.apps:
        th = threading.Thread(target=run_app, args=(app, sites[app], tests, run_dir, args, skip), daemon=True)
        threads.append(th)
        th.start()
        if not args.parallel:
            th.join()
    for th in threads:
        th.join()
    elapsed = round(time.time() - t0, 1)
    with open(run_dir / "timing.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"session_started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                            "wall_secs": elapsed, "apps": args.apps, "parallel": args.parallel,
                            "judge": {k: judge_state.get(k) for k in ("status", "model", "error")}}) + "\n")
    log("runner", "finished in %.1f min. Next: python qa_report.py %s" % (elapsed / 60, run_dir))
    if args.report:
        import qa_report
        qa_report.main([str(run_dir)])
    return 0


def cmd_login(args):
    sites = qa_sites.load_sites(BASE)
    drivers = []
    for app in args.apps:
        d = make_driver(app, args)
        d.get(sites[app]["url"])
        drivers.append(d)
        log(app, "window open. %s" % sites[app]["setup"])
    input("\nSign in to each site in its window (and change the settings noted above). "
          "Press Enter here when done... ")
    for d in drivers:
        d.quit()
    print("Profiles saved in %s. They stay signed in for later runs." % PROFILES)
    return 0


def cmd_probe(args):
    sites = qa_sites.load_sites(BASE)
    rows = []
    for app in args.apps:
        site = sites[app]
        d = make_driver(app, args)
        row = {"app": app}
        try:
            d.get(site["url"])
            time.sleep(4)
            el, sel = find_first(d, site["input"])
            row["input"] = sel or "NOT FOUND"
            row["blocked"] = blocked_reason(d) if el is None else None
            if el is not None:
                ans, extraction, secs, _ = send_and_wait(d, site, app, "Reply with just the word OK.", args)
                _els, rsel = find_all(d, site["response"])
                row.update(response=rsel or "NOT FOUND", extraction=extraction, secs=secs,
                           answer=ans[:40], ok="ok" in ans.lower())
        except Exception as e:
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:150])
        finally:
            d.quit()
        rows.append(row)
        log(app, json.dumps(row))
    good = sum(1 for r in rows if r.get("ok"))
    print("\n%d of %d sites answered and the answer was read correctly." % (good, len(rows)))
    return 0 if good == len(rows) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps", nargs="+", default=None, choices=ALL_APPS,
                    help="apps to test (default: all four; on --resume: the run's own apps)")
    ap.add_argument("--plan", choices=["core", "3d", "full"], default="3d")
    ap.add_argument("--seed", type=int, help="augmentation seed (default from test data: 187)")
    ap.add_argument("--repeat", type=int, default=3, help="times each test runs per app (default 3)")
    ap.add_argument("--filter", help="regex on test_id, e.g. 'QA-03' or 'CX02'")
    ap.add_argument("--max-tests", type=int, help="only the first N tests of the plan")
    ap.add_argument("--run-id")
    ap.add_argument("--resume", help="results/<run_id> folder of an interrupted run")
    ap.add_argument("--redo-body-diff", action="store_true",
                    help="with --resume, also run again the executions whose answer was read from page text")
    ap.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True,
                    help="run the apps at the same time, one Chrome window each (default on)")
    ap.add_argument("--delay", type=float, default=3, help="seconds between messages (default 3)")
    ap.add_argument("--timeout", type=float, default=180, help="max seconds per answer (default 180)")
    ap.add_argument("--stable", type=float, default=4, help="answer is done after N seconds unchanged (default 4)")
    ap.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True,
                    help="second opinion from Ollama (default on, skipped if Ollama is not running)")
    ap.add_argument("--judge-model", default=qa_judge.DEFAULT_MODEL)
    ap.add_argument("--judge-url", default=qa_judge.DEFAULT_URL)
    ap.add_argument("--pause", action=argparse.BooleanOptionalAction, default=True,
                    help="when a site asks you to sign in or pass a check, wait up to 10 min for you "
                         "to do it in that window (default on)")
    ap.add_argument("--tier", nargs="*", help="record the account tier, e.g. chatgpt=Free claude=Pro")
    ap.add_argument("--headless", action="store_true", help="not recommended, sites block headless Chrome")
    ap.add_argument("--profiles-dir", help="folder with the saved Chrome profiles (default chrome_profiles/ here; "
                                           "use ../chrome_profiles to share sign-ins with Martin's script)")
    ap.add_argument("--chrome-binary")
    ap.add_argument("--chromedriver", help="path to a chromedriver (default: Selenium finds one)")
    ap.add_argument("--chrome-arg", action="append", help="extra Chrome flag, e.g. --chrome-arg=--lang=en")
    ap.add_argument("--report", action="store_true", help="build the statistics report when the run ends")
    ap.add_argument("--login", action="store_true", help="open each site to sign in once")
    ap.add_argument("--probe", action="store_true", help="check page elements and read one short answer")
    args = ap.parse_args(argv)
    if not args.apps and not args.resume:
        args.apps = list(ALL_APPS)
    if args.login:
        return cmd_login(args)
    if args.probe:
        return cmd_probe(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
