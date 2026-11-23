"""Regression checks for capture/quality separation and raw-file compatibility."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import domain_knowledge_test as collector
import evaluate_domain_results as evaluator
from domain_capture import capture_error
from test_data import DOMAIN_TESTS


class DomainKnowledgeTests(unittest.TestCase):
    def setUp(self):
        test = DOMAIN_TESTS[6]
        self.record = dict(app="gemini", test_id=test["id"], prompt=test["prompt"],
                           expected=test["expected"], response="Try restarting your router.")

    def test_ui_capture_bypasses_judge_and_is_not_model_fail(self):
        self.record["response"] = "Sign in to connect to Google apps, create images, and more\nSign in\nAuto"
        with patch.object(evaluator, "ollama_grade") as judge:
            row = evaluator.evaluate_record(self.record, "llama3.1:8b", [])
        judge.assert_not_called()
        self.assertEqual(row["result"], "AUTOMATION_ERROR")
        self.assertEqual(evaluator.summarize([row])[-1]["fail"], 0)

    def test_real_answer_with_login_tip_is_not_capture_error(self):
        self.assertIsNone(capture_error("SNMP is Simple Network Management Protocol.\nSign in"))
        self.assertIsNotNone(capture_error("   "))

    def test_judge_outage_and_invalid_output_do_not_pass_or_fail_model(self):
        for error in (ConnectionError("offline"), ValueError("invalid judge JSON")):
            with patch.object(evaluator, "ollama_grade", side_effect=error):
                row = evaluator.evaluate_record(self.record, "llama3.1:8b", [])
            self.assertEqual(row["result"], "REVIEW")
            self.assertTrue(row["evaluation_error"])

    def test_router_rubric_is_only_sent_for_router_test(self):
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps({"message": {"content": json.dumps(
            {"verdict": "PASS", "reasoning": "Meets expected behavior"})}}).encode()
        for test in (DOMAIN_TESTS[0], DOMAIN_TESTS[6]):
            rec = {**self.record, "test_id": test["id"], "prompt": test["prompt"], "expected": test["expected"]}
            with patch.object(evaluator.urllib.request, "urlopen", return_value=reply) as request:
                evaluator.ollama_grade(rec, "llama3.1:8b")
            body = json.loads(request.call_args.args[0].data)
            self.assertEqual("ambiguous router question" in body["messages"][0]["content"], test["id"] == "TC-DK-07")

    def test_invalid_judge_verdict_requires_review(self):
        reply = Mock()
        reply.__enter__ = Mock(return_value=reply)
        reply.__exit__ = Mock(return_value=False)
        reply.read.return_value = json.dumps({"message": {"content": json.dumps(
            {"verdict": "MAYBE", "reasoning": "Unsure"})}}).encode()
        with patch.object(evaluator.urllib.request, "urlopen", return_value=reply):
            row = evaluator.evaluate_record(self.record, "llama3.1:8b", [])
        self.assertEqual(row["result"], "REVIEW")
        self.assertIn("Invalid judge output", row["evaluation_error"])

    def test_review_is_hash_bound_and_retains_original_judge_result(self):
        review = {"app": "gemini", "test_id": "TC-DK-07",
                  "response_sha256": hashlib.sha256(self.record["response"].encode()).hexdigest(),
                  "result": "PARTIAL", "reason": "Missing initial clarification."}
        with patch.object(evaluator, "ollama_grade", return_value=("PASS", "Useful", {})):
            row = evaluator.evaluate_record(self.record, "llama3.1:8b", [review])
            self.assertEqual((row["result"], row["judge_result"]), ("PARTIAL", "PASS"))
            changed = {**self.record, "response": "What symptoms are you seeing?"}
            self.assertFalse(evaluator.evaluate_record(changed, "llama3.1:8b", [review])["review_applied"])

    def test_schema_accepts_manual_claude_and_rejects_duplicate_or_wrong_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claude.json"
            rows = [dict(app="claude", test_id=t["id"], prompt=t["prompt"],
                         expected=t["expected"], response="Manual response") for t in DOMAIN_TESTS]
            path.write_text(json.dumps(rows))
            self.assertEqual(len(evaluator.load_records(Path(directory))), 8)
            for field, value in (("test_id", "TC-DK-01"), ("prompt", "Wrong prompt")):
                invalid = copy.deepcopy(rows)
                invalid[1][field] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    evaluator.load_records(Path(directory))

    def test_capture_errors_are_excluded_from_model_quality_denominator(self):
        rows = [{"app": "gemini", "result": "PASS", "judge_result": "PASS"},
                {"app": "gemini", "result": "PARTIAL", "judge_result": "PARTIAL"},
                {"app": "gemini", "result": "AUTOMATION_ERROR", "judge_result": ""}]
        total = evaluator.summarize(rows)[-1]
        self.assertEqual(total["full_pass_rate"], "50.0%")
        self.assertEqual(total["semantic_evaluations"], 2)

    def test_gemini_never_captures_body_fallback(self):
        with patch.object(collector, "visible_texts", return_value=[]), \
             patch.object(collector, "body_text") as body, \
             patch.object(collector.time, "time", side_effect=[0, 1, 3]), \
             patch.object(collector.time, "sleep"):
            with self.assertRaises(collector.TimeoutException):
                collector.wait_response(None, collector.APP_CONFIG["gemini"], [], "", "prompt", timeout=2)
        body.assert_not_called()

    def test_targeted_retry_keeps_other_rows_and_archives_original(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            original = [dict(app="gemini", test_id=t["id"], prompt=t["prompt"],
                             expected=t["expected"], response="original " + t["id"]) for t in DOMAIN_TESTS]
            path = raw / "gemini.json"
            path.write_text(json.dumps(original))
            before = path.read_bytes()
            with patch.object(collector, "ROOT", Path(directory)), \
                 patch.object(collector, "RAW_DIR", raw), \
                 patch.object(collector, "find_first"), patch.object(collector, "enter_prompt"), \
                 patch.object(collector, "submit"), patch.object(collector, "visible_texts", return_value=[]), \
                 patch.object(collector, "body_text", return_value=""), \
                 patch.object(collector.time, "sleep"), \
                 patch.object(collector, "wait_response", side_effect=["Real FCAPS answer", collector.TimeoutException("blocked")]):
                (Path(directory) / "results").mkdir()
                driver = Mock()
                failures = collector.run_app(driver, "gemini", ["TC-DK-03", "TC-DK-08"], True)
            after = json.loads(path.read_text())
            self.assertEqual(failures, 1)
            self.assertEqual(after[2]["response"], "Real FCAPS answer")
            for index in [0, 1, 3, 4, 5, 6, 7]:
                self.assertEqual(after[index], original[index])
            self.assertEqual(next((raw / "archive").glob("*.json")).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
