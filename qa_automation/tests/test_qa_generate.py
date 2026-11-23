"""Checks on test generation and data augmentation (repeatable, consistent, correct)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qa_generate as g  # noqa: E402

MODEL = g.load("qa_model_3d.json")
DATA = g.load("qa_test_data.json")


def test_data_is_consistent():
    assert g.validate(MODEL, DATA) == []


def test_generation_is_repeatable():
    a = g.build_suite(MODEL, DATA, "3d")
    b = g.build_suite(MODEL, DATA, "3d")
    assert a == b
    assert a["meta"]["content_sha256"] == b["meta"]["content_sha256"]


def test_seed_changes_the_augmented_values():
    a = g.build_suite(MODEL, DATA, "3d", seed=1)
    b = g.build_suite(MODEL, DATA, "3d", seed=2)
    assert a["meta"]["content_sha256"] != b["meta"]["content_sha256"]


def test_plan_sizes():
    assert g.build_suite(MODEL, DATA, "core")["meta"]["n_tests"] == 8
    assert g.build_suite(MODEL, DATA, "3d")["meta"]["n_tests"] == 36
    assert g.build_suite(MODEL, DATA, "full")["meta"]["n_tests"] == 56


def test_every_cell_of_the_3d_model_is_generated():
    suite = g.build_suite(MODEL, DATA, "3d")
    cells = {(t["input_class"], t["context_class"]) for t in suite["tests"]}
    assert len(cells) == len(MODEL["input"]["classes"]) * len(MODEL["context"]["classes"])


def test_noise_keeps_technical_tokens():
    for row in g.build_inputs(MODEL, DATA):
        if row["augmentation"] != "NOISE":
            continue
        base = next(t for t in DATA["tests"] if t["id"] == row["base_id"])
        for tok in base["prompt"].split():
            core = tok.strip(",.?")
            if any(c.isdigit() for c in core) or any(c.isupper() for c in core[1:]):
                assert core in row["prompt"], (core, row["prompt"])
        assert row["prompt"] != base["prompt"]


def test_param_variants_recompute_the_oracle():
    rows = {r["input_id"]: r for r in g.build_inputs(MODEL, DATA)}
    p3 = rows["QA-03-PARAM"]["oracle"]
    assert p3["first"] > 2 ** 31 and p3["second"] < 2 ** 32 - p3["first"] + 10 ** 7
    assert "{:,}".format(p3["first"]) in rows["QA-03-PARAM"]["prompt"]
    p7 = rows["QA-07-PARAM"]["oracle"]
    assert p7["ip"] in rows["QA-07-PARAM"]["prompt"] and p7["community"] in rows["QA-07-PARAM"]["prompt"]
    assert rows["QA-08-PARAM"]["oracle"]["subject"] in rows["QA-08-PARAM"]["prompt"]
