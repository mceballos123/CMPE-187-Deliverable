# Question/Answering test automation (CMPE 187, Group 8, Deliverable 3)

Owner: Joel Moyal. Feature: Question/Answering (accuracy, relevance and completeness of
answers to SNMP / network management questions) on ChatGPT, Claude, Gemini and Perplexity.

The scripts follow the 3D test model from Deliverable 2:

| Dimension | Classes | Where it lives |
|---|---|---|
| Context | 4 apps x QA-CX-01 fresh session, QA-CX-02 carry-over session (Martin's 3 distractor prompts first) | `data/qa_model_3d.json` |
| Input | QA-IC-01 to QA-IC-08, each expanded by augmentation (ORIG, PARA, NOISE, PARAM) | `data/qa_test_data.json` |
| Output | QA-OC-01 to QA-OC-07, assigned to every answer by the oracle | `qa_oracle.py` |

## Results in this repo

Two real runs of the `3d` plan (36 test cases, 2 repeats), both logged out on free tiers.
The combined report is `results/run_20261122_201114/report/summary.md`.

| App | Graded | Passed | Run |
|---|---|---|---|
| ChatGPT | 45 of 72 | 45 | `results/run_20261122_201114` (Chrome on a Mac) |
| Claude | 0 of 72 | - | `results/run_20261122_201114` (Chrome on a Mac) |
| Gemini | 72 of 72 | 70 | `results/run_cloud_gemini_perplexity` (Chrome on a cloud machine) |
| Perplexity | 72 of 72 | 71 | `results/run_cloud_gemini_perplexity` (Chrome on a cloud machine) |
| Total | 189 of 288 | 186 (98.4%) | |

Why some executions did not run (also in the report table "Executions not run, and why"):

- **Claude:** claude.ai showed a "verify you are human" check to the Selenium-controlled
  browser on the very first execution. `results.jsonl` records it as `"status": "blocked",
  "error": "challenge"`. The scripts never try to get around such checks, so the Chrome window
  was closed, and the other 71 Claude executions are recorded as errors ("browser window was
  closed"). None of them has a verdict and none counts in the pass rate. This is a finding in
  itself: Claude's web app cannot be tested this way without signing in through a normal browser.
- **ChatGPT:** logged out, ChatGPT allowed 45 executions and then showed "You have reached the
  anonymous message limit" (screenshot: `results/run_20261122_201114/evidence/chatgpt_message_limit.png`).
  The runner recognized the message (`"status": "blocked", "error": "limit"`) and stopped ChatGPT
  cleanly, so 26 executions never started. They can be finished after signing in with
  `python qa_runner.py --resume results/run_20261122_201114 --apps chatgpt`.

## Files

| File | What it does |
|---|---|
| `data/qa_model_3d.json` | 3D model: requirements, context/input/output classes, decision rules R1-R8, run plans |
| `data/qa_test_data.json` | TC-QA-01 to TC-QA-08: prompts, paraphrases, reference key points, oracle settings |
| `qa_generate.py` | Data discovery (`discover`), data augmentation (`augment`), test case generation (`generate`) |
| `qa_oracle.py` | Automatic validation: maps each answer to an output class and a verdict |
| `qa_judge.py` | Second opinion from a local Ollama model, REVIEW on disagreement, re-grading of saved runs |
| `qa_sites.py` | URLs and CSS selectors for the four sites (override in `sites_override.json`) |
| `qa_runner.py` | Selenium execution: login, probe, runs with repeats, resume |
| `qa_review.py` | Decide REVIEW items by hand (kept separate from the raw results) |
| `qa_report.py` | Statistics: results, repeatability, cost, script complexity, coverage, charts |
| `tests/` | 54 unit tests for the oracle, the generator and the report (`python -m pytest -q tests`) |

## Fastest way: one command

```bash
cd qa_automation
./run_my_part.sh demo     # short run for the video (8 tests, ChatGPT + Perplexity)
./run_my_part.sh          # full run for the report (3d plan, 2 repeats, ChatGPT + Claude)
```

Gemini and Perplexity were already run (3d plan, 2 repeats, logged out, headless Chrome) and
their results are in `results/run_cloud_gemini_perplexity/`. ChatGPT shows a human check to
automated browsers without a normal profile and Claude needs a sign-in, so those two run on your
Mac. The final report merges both runs. `APPS="chatgpt claude gemini perplexity" ./run_my_part.sh`
runs all four again.

It sets up Python, runs discovery, augmentation, generation and the self-tests, opens the
Chrome windows for you to sign in (first time only), probes the sites, runs the tests,
asks you to decide any REVIEW items, and builds the report. At the end it prints the path of
`report.zip`, which is the file to send back. If a site hits its message limit, run the
`--resume` command it prints later.

## Setup (once)

```bash
cd qa_automation                      # put this folder in the root of the group repo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b               # optional judge, same model as the memory tests
python qa_generate.py discover        # checks data, model, Chrome, profiles, Ollama
python qa_runner.py --login           # 4 Chrome windows open: sign in, set memory off, press Enter
python qa_runner.py --probe           # sends "Reply with just the word OK." to each site
```

Needs Python 3.9+ and Google Chrome. Selenium finds the matching ChromeDriver by itself.
The Chrome profiles are saved in `chrome_profiles/` and are in `.gitignore`. Never commit them,
they hold your logins.

## Runs

```bash
python qa_runner.py --plan core --apps chatgpt --repeat 1           # 8 executions, good for the video
python qa_runner.py --plan 3d --repeat 3 --tier chatgpt=Free claude=Free gemini=Free perplexity=Free
python qa_runner.py --resume results/run_YYYYMMDD_HHMMSS            # after a limit or a crash
python qa_runner.py --resume results/run_YYYYMMDD_HHMMSS --apps chatgpt   # finish one app only
python qa_review.py results/run_YYYYMMDD_HHMMSS                     # decide REVIEW items
python qa_report.py results/run_YYYYMMDD_HHMMSS --dev-hours 12      # statistics + charts
```

Plan sizes (executions per repeat, 4 apps): `core` 32, `3d` 144, `full` 224.
With 4 apps in parallel, the `3d` plan takes roughly 30 to 45 minutes per repeat.

Free tiers have message limits. When a site shows a limit, that app stops and the others keep
going. Wait, then use `--resume` with the same folder: finished executions are skipped.
If a site shows a sign-in page or a human check, the runner waits up to 10 minutes for you to
deal with it in that window and then carries on by itself. It never tries to get around those checks.
Optional pop-ups (cookie banners, "log in for more features") are closed with their "Not now" /
"Only necessary" button. If a site answers with its own error message ("Something went wrong"),
the execution is retried once and never graded as a wrong answer.

## Working next to Martin's Intelligent Memory scripts

Put this folder in the root of `mceballos123/CMPE-187-Deliverable`, next to `llm_memory_test.py`.
The two share the same design: Selenium on the web apps, a keyword/rule grader, and the
`llama3.2:3b` Ollama judge with REVIEW when they disagree. The site selectors here are merged
with his, and the carry-over context (QA-CX-02) uses his three distractor prompts
(OSI layers, VLAN, router), so both features are tested under the same conditions.

His script keeps its Chrome profiles in `../chrome_profiles/`. To reuse those sign-ins:
`python qa_runner.py --profiles-dir ../chrome_profiles ...` (do not run both scripts at the
same time on the same app: one Chrome profile can only be open once).

## If something goes wrong

- **Google says the browser may not be secure (Gemini sign-in):** Google blocks sign-in in a browser that
  Selenium controls. Open a normal Chrome on the same profile, sign in there, close it, then run as usual:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --user-data-dir="$PWD/chrome_profiles/gemini"`
- **"Chrome did not start for <app>":** another run or Chrome window is using that profile. Close it.
- **Probe says NOT FOUND for input or response:** the site changed its page. Inspect the element in
  DevTools and add the new CSS selector to `sites_override.json`, e.g.
  `{"claude": {"response": ["div.new-answer-class"]}}`. No code change needed.
- **An app hits its message limit:** that app stops, the others continue. Run `--resume` later.

## How an answer is graded

1. `qa_oracle.py` checks the answer against the key points of its test case with fixed rules
   (regex, number parsing with a 1% tolerance, net-snmp command parsing). It flags wrong values
   (QA-OC-05) and invented RFCs or options (QA-OC-06), and gives one output class.
2. If Ollama is running, `qa_judge.py` asks `llama3.2:3b` (temperature 0, fixed seed) to pick an
   output class from the same key points.
3. Same verdict from both: that is the final verdict. Different verdicts: REVIEW, decided by a
   person with `qa_review.py`. No Ollama: oracle only, and the report says so.

The oracle is tested on 41 hand-written answers (good, partial, wrong, fabricated) in
`tests/test_qa_oracle.py`. Every output class is reached at least once.

## Repeatability

- Same seed, same suite: `generate` writes a sha256 of the suite. Run it twice and it matches.
- Every execution opens a fresh chat. ChatGPT uses a temporary chat (memory off).
- `--repeat N` runs every test N times. The report shows how often the verdict stayed the same.
- `config.json` in each run folder records the suite, script hashes, Python/Selenium versions
  and the account tiers, so a run can be reproduced or compared later.
- `python qa_judge.py regrade results/<run>` re-grades saved answers without sending anything.

## Video demo script (about 5 minutes)

1. `python qa_generate.py discover`: data discovery and setup checks.
2. `python qa_generate.py augment`: show ORIG, PARA, NOISE and PARAM prompts, and point out the
   recomputed expected answer for QA-03-PARAM and QA-07-PARAM.
3. `python qa_generate.py generate --plan 3d`: the 3D coverage matrix (16 of 16 cells, 144 executions).
4. `python -m pytest -q tests`: the validator is checked before we trust it.
5. `python qa_runner.py --plan core --apps chatgpt perplexity --repeat 1`: show the Chrome windows
   typing, the live log with verdicts and output classes.
6. Open `results/<run>/answers/` and one screenshot in `screens/`.
7. `python qa_report.py results/<run>`: open `report/summary.md` and the two charts.
