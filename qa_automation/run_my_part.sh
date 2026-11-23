#!/bin/bash
# Joel's part of CMPE 187 Deliverable 3 (Question/Answering), start to finish.
#
#   ./run_my_part.sh          full run: 3d plan, 2 repeats (about 1 hour)
#                             apps: ChatGPT and Claude by default, because Gemini and Perplexity were
#                             already run (results/run_cloud_gemini_perplexity). APPS="chatgpt claude gemini perplexity" runs all 4.
#   ./run_my_part.sh demo     short run for the video: core plan, ChatGPT + Perplexity, 1 repeat
#
# Optional:  REPEAT=3 ./run_my_part.sh      DEV_HOURS=12 ./run_my_part.sh
#            RUNNER_ARGS="--profiles-dir ../chrome_profiles" ./run_my_part.sh   (reuse Martin's sign-ins)
set -e
cd "$(dirname "$0")"
MODE="${1:-full}"
if [ "$MODE" = "demo" ]; then RUN_APPS="${APPS:-chatgpt perplexity}"; else RUN_APPS="${APPS:-chatgpt claude}"; fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo; echo "== 1. Data discovery and setup"
python qa_generate.py discover
echo; echo "== 2. Data augmentation"
python qa_generate.py augment
echo; echo "== 3. Test case generation from the 3D model"
python qa_generate.py generate --plan 3d
echo; echo "== 4. Checking the validator"
python -m pytest -q tests

if [ ! -d chrome_profiles ] && [[ "$RUNNER_ARGS" != *--profiles-dir* ]]; then
  echo; echo "== Sign in once in each Chrome window (memory off), then press Enter here"
  python qa_runner.py --login --apps $RUN_APPS $RUNNER_ARGS
fi

echo; echo "== 5. Probe: can every site be reached and read?"
python qa_runner.py --probe --apps $RUN_APPS $RUNNER_ARGS || echo ">>> Some sites need attention (see above). The run still tries them."

echo; echo "== 6. Test execution"
if [ "$MODE" = "demo" ]; then
  python qa_runner.py --plan core --apps $RUN_APPS --repeat 1 $RUNNER_ARGS
else
  python qa_runner.py --plan 3d --repeat "${REPEAT:-2}" --apps $RUN_APPS --tier chatgpt=Free claude=Free gemini=Free perplexity=Free $RUNNER_ARGS
fi
RUN=$(ls -td results/run_* | head -1)

echo; echo "== 7. Decide REVIEW items (oracle and judge disagreed)"
python qa_review.py "$RUN"

echo; echo "== 8. Statistics report"
EARLIER=""
if [ "$MODE" != "demo" ]; then
  for d in results/run_cloud_*; do [ -d "$d" ] && [ "$d" != "$RUN" ] && EARLIER="$EARLIER $d"; done
fi
python qa_report.py "$RUN" $EARLIER ${DEV_HOURS:+--dev-hours "$DEV_HOURS"}
(cd "$RUN" && zip -qr report.zip report config.json results.jsonl answers)
echo; echo "Done. Results and report: $RUN/report.zip"
echo "If a site hit its limit, finish with: python qa_runner.py --resume $RUN"
