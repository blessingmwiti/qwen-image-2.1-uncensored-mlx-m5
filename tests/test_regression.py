"""Regression: recorded artifacts + metrics schema (fast, no model load).

Guards the known-good baseline: hashes, metric files, eval-suite frozenness.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def test_baseline_hashes_recorded():
    r = _load("experiments/baseline/results.json")
    assert r["revision"] == "790c92633540aa0cb11d9abf19eb46d861714758"
    assert r["status"] == "success"
    assert r["total_gb"] > 30
    assert len(r["files"]) == 7
    assert all(len(f["sha256"]) == 64 for f in r["files"])


def test_smoke_metrics_schema():
    for p in ("experiments/baseline/smoke_s8.json", "experiments/baseline/smoke_s40.json"):
        m = _load(p)
        for k in ("model", "revision", "resolution", "steps", "seed", "generation_seconds", "sha_prefix", "status"):
            assert k in m, (p, k)
        assert m["status"] == "success"
        assert m["resolution"] == "1024x1024"


def test_exp001_bar():
    r = _load("experiments/exp-001/results.json")
    assert r["status"] == "success"
    assert len(r["probes"]) == 4


def test_exp002_bar():
    r = _load("experiments/exp-002/results.json")
    assert r["status"] == "success"
    assert r["v2_cosine"] < 0.9  # hook dominates
    assert r["v3_l2"] == 0.0  # empty-guard exact


def test_hybrid_metrics_schema():
    for p in ("experiments/hybrid/metrics_hybrid_q8.json", "experiments/hybrid/metrics_hybrid_q4.json"):
        m = _load(p)
        assert m["status"] == "success"
        assert m["seed"] == 42 and m["steps"] in (8, 40)
        assert len(m["sha_prefix"]) == 16


def test_eval_suite_frozen():
    s = _load("tests/prompts/suite.json")
    assert s["eval_suite_version"] == "v1"
    assert set(s["categories"]) == {"benign", "artistic", "text-rendering", "composition", "editing", "difficult", "safety-behavior"}
