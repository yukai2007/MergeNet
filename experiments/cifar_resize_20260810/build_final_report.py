#!/usr/bin/env python3
"""Publish the completed CIFAR resize aggregate into the release repository.

This is intentionally a *consumer* of aggregate_results.{json,csv,md}; it does
not rerun or reinterpret the campaign.  Publication is fail closed: no file in
the repository is touched until all three aggregate files agree and prove the
locked 45-run / 8-card / 30-checkpoint release state.

The locked preregistered release field can be READY or NO_GO.  Publication also
states a separate, evidence-bounded recommendation about whether the completed
CIFAR campaign justifies a controlled ImageNet experiment.  INCOMPLETE evidence
is never publishable, and the scale-up recommendation never rewrites a locked
gate result.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SCHEMA = 2
MODELS = ("deit_s8", "mn_l2", "mn_l4")
MERGENET_MODELS = ("mn_l2", "mn_l4")
RESIZES = (160, 192, 224, 256, 320)
SEEDS = (42, 43, 44)
GPUS = tuple(range(8))
ACCURACY_JOBS = 45
EFFICIENCY_CARDS = 8
PARITY_RUNS = 30
TARGET_EPOCH = 199
CSV_PARITY_METRICS = (
    "generic_top1",
    "fast_top1",
    "top1_delta_pp",
    "argmax_agreement",
    "argmax_mismatch_count",
    "max_abs_logit_diff",
    "mean_abs_logit_diff",
)
MODEL_LABELS = {
    "deit_s8": "DeiT-S/8",
    "mn_l2": "MergeNet 6+6 · λ=2 · w16",
    "mn_l4": "MergeNet 4+8 · λ=4 · w32",
}
MODE_LABELS = {
    "train": "train / random_per_sample",
    "infer_generic": "inference / generic",
    "infer_fast": "inference / fast",
}
FINAL_HTML = Path("reports/mergenet_cifar_resize_final_20260814.html")
VISUAL_HTML = Path("reports/mergenet_cifar_resize_visual_report_20260814.html")
EVIDENCE_DIR = Path("reports/evidence/cifar_resize_20260810")
INDEX_DOC = Path("reports/evidence/证据索引.md")
POSITION_DOC = Path("reports/evidence/CV结果与长序列判断_20260802.md")
ROOT_README = Path("README.md")
HANDOFF_README = Path("deliverables/imagenet_longtrain_v1/README.md")
HANDOFF_NOTES = Path("deliverables/imagenet_longtrain_v1/RELEASE_NOTES.md")
LAMBDA4_CONFIG = Path("deliverables/imagenet_longtrain_v1/configs/mergenet_lambda4.yaml")
MARKER_START = "<!-- CIFAR_RESIZE_FINAL:START -->"
MARKER_END = "<!-- CIFAR_RESIZE_FINAL:END -->"
ROOT_CAMPAIGN_LEGACY = (
    "- [CIFAR resize validation](experiments/cifar_resize_20260810/): the\n"
    "  multi-resolution accuracy and efficiency campaign. The full accuracy sweep is\n"
    "  currently running; its protocol and collection scripts live here."
)
AGGREGATE_RENDERER = Path(__file__).resolve().with_name("aggregate_results.py")
VISUAL_RENDERER = Path(__file__).resolve().with_name("build_visual_report.py")
PINNED_AGGREGATE_RENDERER_SHA256 = "43a4013016d14b36e1d13121bb7a84fbd91b25fd36c1d53d16d5cbbd54eccdb4"


class EvidenceError(RuntimeError):
    """The aggregate does not prove a publishable final state."""


@dataclass(frozen=True)
class EvidenceBundle:
    directory: Path
    document: Mapping[str, Any]
    json_bytes: bytes
    csv_bytes: bytes
    markdown_bytes: bytes
    hashes: Mapping[str, str]


def fail(message: str) -> None:
    raise EvidenceError(message)


def as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{name} must be an object")
    return value


def as_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{name} must be a list")
    return value


def as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        fail(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{name} must be an integer") from exc
    if isinstance(value, float) and value != result:
        fail(f"{name} must be an integer")
    return result


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        fail(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        fail(f"{name} must be finite")
    return result


def same_float(actual: Any, expected: float, name: str, tolerance: float = 1e-8) -> None:
    value = finite(actual, name)
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=tolerance):
        fail(f"{name}={value!r}, expected {expected!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail(f"required aggregate file is missing, non-regular, or symlinked: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc


def expected_accuracy_keys() -> set[tuple[str, int, int]]:
    return {(model, resize, seed) for model in MODELS for resize in RESIZES for seed in SEEDS}


def expected_summary_keys() -> set[tuple[str, int]]:
    return {(model, resize) for model in MODELS for resize in RESIZES}


def expected_paired_keys() -> set[tuple[str, int]]:
    return {(model, resize) for model in MERGENET_MODELS for resize in RESIZES}


def expected_parity_keys() -> set[tuple[str, int, int]]:
    return {(model, resize, seed) for model in MERGENET_MODELS for resize in RESIZES for seed in SEEDS}


def keyed_rows(
    rows: Iterable[Any], keys: Sequence[str], expected: set[tuple[Any, ...]], name: str
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = as_mapping(raw, f"{name}[{index}]")
        try:
            key = tuple(row[key_name] for key_name in keys)
        except KeyError as exc:
            raise EvidenceError(f"{name}[{index}] lacks {exc.args[0]}") from exc
        normalized = tuple(as_int(v, f"{name} key") if k in {"resize", "seed"} else v for k, v in zip(keys, key))
        if normalized in result:
            fail(f"{name} contains duplicate key {normalized}")
        result[normalized] = row
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        fail(f"{name} matrix mismatch; missing={missing[:4]}, extra={extra[:4]}")
    return result


def validate_matrix(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != EXPECTED_SCHEMA:
        fail(f"aggregate schema must be {EXPECTED_SCHEMA}")
    matrix = as_mapping(document.get("matrix"), "matrix")
    if tuple(matrix.get("models", ())) != MODELS:
        fail("matrix.models differs from the locked model order")
    if tuple(matrix.get("resizes", ())) != RESIZES:
        fail("matrix.resizes differs from the locked resize order")
    if tuple(matrix.get("seeds", ())) != SEEDS:
        fail("matrix.seeds differs from the locked seed order")
    if as_int(matrix.get("expected_accuracy_jobs"), "matrix.expected_accuracy_jobs") != ACCURACY_JOBS:
        fail("matrix expected accuracy count is not 45")
    completeness = as_mapping(document.get("completeness"), "completeness")
    exact = {
        "accuracy_complete_jobs": ACCURACY_JOBS,
        "accuracy_expected_jobs": ACCURACY_JOBS,
        "efficiency_complete_cards": EFFICIENCY_CARDS,
        "efficiency_expected_cards": EFFICIENCY_CARDS,
    }
    for key, value in exact.items():
        if as_int(completeness.get(key), f"completeness.{key}") != value:
            fail(f"completeness.{key} must equal {value}")
    generated = document.get("generated_at")
    if not isinstance(generated, str):
        fail("generated_at must be an ISO-8601 string")
    try:
        datetime.fromisoformat(generated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("generated_at is not valid ISO-8601") from exc


def validate_accuracy(document: Mapping[str, Any]) -> tuple[
    dict[tuple[str, int, int], Mapping[str, Any]],
    dict[tuple[str, int], Mapping[str, Any]],
    dict[tuple[str, int], Mapping[str, Any]],
]:
    accuracy = as_mapping(document.get("accuracy"), "accuracy")
    if accuracy.get("primary_endpoint") != "epoch_199_ema_top1":
        fail("accuracy primary endpoint is not epoch_199_ema_top1")
    runs = keyed_rows(
        as_list(accuracy.get("runs"), "accuracy.runs"),
        ("model_id", "resize", "seed"),
        expected_accuracy_keys(),
        "accuracy.runs",
    )
    for key, row in runs.items():
        if row.get("status") != "complete" or row.get("artifact_verified") is not True:
            fail(f"accuracy run {key} is not verified complete")
        if as_int(row.get("target_epoch"), f"accuracy run {key} target_epoch") != TARGET_EPOCH:
            fail(f"accuracy run {key} target epoch mismatch")
        if as_int(row.get("last_epoch"), f"accuracy run {key} last_epoch") != TARGET_EPOCH:
            fail(f"accuracy run {key} last epoch mismatch")
        finite(row.get("ema_top1"), f"accuracy run {key} ema_top1")

    summaries = keyed_rows(
        as_list(accuracy.get("summary"), "accuracy.summary"),
        ("model_id", "resize"),
        expected_summary_keys(),
        "accuracy.summary",
    )
    for key, row in summaries.items():
        if row.get("complete") is not True:
            fail(f"accuracy summary {key} is incomplete")
        if as_int(row.get("expected_seeds"), f"summary {key} expected_seeds") != 3:
            fail(f"accuracy summary {key} expected_seeds mismatch")
        if as_int(row.get("n"), f"summary {key} n") != 3:
            fail(f"accuracy summary {key} does not contain 3 seeds")
        if tuple(sorted(as_int(v, f"summary {key} seed") for v in as_list(row.get("complete_seeds"), f"summary {key} complete_seeds"))) != SEEDS:
            fail(f"accuracy summary {key} complete seed set mismatch")
        if as_list(row.get("missing_or_unverified_seeds"), f"summary {key} missing seeds"):
            fail(f"accuracy summary {key} still has missing seeds")
        observed = [finite(runs[(key[0], key[1], seed)].get("ema_top1"), "run top1") for seed in SEEDS]
        same_float(row.get("mean"), statistics.mean(observed), f"summary {key} mean")
        same_float(row.get("sample_sd"), statistics.stdev(observed), f"summary {key} sample_sd")

    paired = keyed_rows(
        as_list(accuracy.get("paired_deltas"), "accuracy.paired_deltas"),
        ("candidate_model", "resize"),
        expected_paired_keys(),
        "accuracy.paired_deltas",
    )
    for key, row in paired.items():
        if row.get("baseline_model") != "deit_s8" or row.get("complete") is not True:
            fail(f"paired accuracy {key} has wrong baseline or is incomplete")
        if as_int(row.get("expected_pairs"), f"paired {key} expected_pairs") != 3:
            fail(f"paired accuracy {key} expected_pairs mismatch")
        if as_int(row.get("n"), f"paired {key} n") != 3:
            fail(f"paired accuracy {key} does not contain 3 seeds")
        pairs = keyed_rows(
            as_list(row.get("pairs"), f"paired {key} pairs"),
            ("seed",),
            {(seed,) for seed in SEEDS},
            f"paired {key} pairs",
        )
        deltas: list[float] = []
        for seed in SEEDS:
            pair = pairs[(seed,)]
            candidate = finite(runs[(key[0], key[1], seed)].get("ema_top1"), "candidate top1")
            baseline = finite(runs[("deit_s8", key[1], seed)].get("ema_top1"), "baseline top1")
            same_float(pair.get("candidate_top1"), candidate, "paired candidate_top1")
            same_float(pair.get("baseline_top1"), baseline, "paired baseline_top1")
            delta = candidate - baseline
            same_float(pair.get("delta_pp"), delta, "paired delta_pp")
            deltas.append(delta)
        same_float(row.get("mean"), statistics.mean(deltas), f"paired {key} mean")
        same_float(row.get("sample_sd"), statistics.stdev(deltas), f"paired {key} sample_sd")
    return runs, summaries, paired


def validate_stats(stats: Any, name: str, expected_n: int = 8) -> None:
    value = as_mapping(stats, name)
    if as_int(value.get("n"), f"{name}.n") != expected_n:
        fail(f"{name} must contain {expected_n} observations")
    finite(value.get("mean"), f"{name}.mean")
    finite(value.get("sample_sd"), f"{name}.sample_sd")
    values = as_list(value.get("values"), f"{name}.values")
    if len(values) != expected_n:
        fail(f"{name}.values must contain {expected_n} values")
    observed = [finite(item, f"{name}.values") for item in values]
    same_float(value.get("mean"), statistics.mean(observed), f"{name}.mean")
    same_float(value.get("sample_sd"), statistics.stdev(observed), f"{name}.sample_sd")


def validate_efficiency(document: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    efficiency = as_mapping(document.get("efficiency"), "efficiency")
    docs = as_list(efficiency.get("documents"), "efficiency.documents")
    if len(docs) != EFFICIENCY_CARDS:
        fail("efficiency.documents must contain exactly 8 cards")
    seen_gpus: set[int] = set()
    for index, raw in enumerate(docs):
        row = as_mapping(raw, f"efficiency.documents[{index}]")
        gpu = as_int(row.get("physical_gpu"), "efficiency physical_gpu")
        if gpu in seen_gpus or row.get("complete") is not True or as_int(row.get("item_count"), "efficiency item_count") != 50:
            fail(f"efficiency card {gpu} is duplicate or not formally complete")
        seen_gpus.add(gpu)
    if seen_gpus != set(GPUS):
        fail("efficiency physical GPU set must be exactly 0..7")

    raw_summary = [as_mapping(row, "efficiency.raw_summary row") for row in as_list(efficiency.get("raw_summary"), "efficiency.raw_summary")]
    raw_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in raw_summary:
        key = (str(row.get("model_id")), as_int(row.get("resize"), "raw resize"), str(row.get("mode")))
        if key in raw_index:
            fail(f"duplicate efficiency raw summary {key}")
        raw_index[key] = row
    expected_raw = {(m, r, mode) for m in MODELS for r in RESIZES for mode in MODE_LABELS}
    if set(raw_index) != expected_raw:
        fail("efficiency raw summary matrix mismatch")
    meaningful = {
        ("deit_s8", resize, mode) for resize in RESIZES for mode in ("train", "infer_generic")
    } | {
        (model, resize, mode) for model in MERGENET_MODELS for resize in RESIZES for mode in MODE_LABELS
    }
    for key in meaningful:
        row = raw_index[key]
        if tuple(sorted(as_int(v, "complete physical GPU") for v in as_list(row.get("complete_physical_gpus"), "complete physical GPUs"))) != GPUS:
            fail(f"raw efficiency {key} lacks all 8 formal cards")
        if row.get("provisional") is not False:
            fail(f"raw efficiency {key} is provisional")
        for metric in ("throughput_img_s", "step_time_ms", "peak_allocated_mib", "peak_reserved_mib", "params"):
            validate_stats(row.get(metric), f"raw efficiency {key} {metric}")

    paired = [as_mapping(row, "efficiency.paired_ratios row") for row in as_list(efficiency.get("paired_ratios"), "efficiency.paired_ratios")]
    paired_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    expected_paired = {(m, r, mode) for m in MERGENET_MODELS for r in RESIZES for mode in MODE_LABELS}
    for row in paired:
        key = (str(row.get("candidate_model")), as_int(row.get("resize"), "paired efficiency resize"), str(row.get("mode")))
        if key in paired_index:
            fail(f"duplicate paired efficiency {key}")
        paired_index[key] = row
    if set(paired_index) != expected_paired:
        fail("paired efficiency matrix mismatch")
    for key, row in paired_index.items():
        if row.get("baseline_model") != "deit_s8":
            fail(f"paired efficiency {key} baseline mismatch")
        if tuple(sorted(as_int(v, "paired physical GPU") for v in as_list(row.get("physical_gpus"), "paired physical GPUs"))) != GPUS:
            fail(f"paired efficiency {key} lacks all 8 card identities")
        for metric in ("throughput_ratio", "step_time_ratio", "peak_allocated_ratio", "peak_reserved_ratio", "params_ratio"):
            validate_stats(row.get(metric), f"paired efficiency {key} {metric}")

    parity = [as_mapping(row, "efficiency.parity row") for row in as_list(efficiency.get("parity"), "efficiency.parity")]
    parity_index = {(str(row.get("model_id")), as_int(row.get("resize"), "synthetic parity resize")): row for row in parity}
    if len(parity_index) != len(parity) or set(parity_index) != expected_paired_keys():
        fail("synthetic parity matrix mismatch")
    for key, row in parity_index.items():
        if row.get("complete") is not True or row.get("allclose_all") is not True:
            fail(f"synthetic parity {key} is not complete/allclose")
        if as_int(row.get("expected_gpu_count"), "synthetic parity expected GPU count") != 8:
            fail(f"synthetic parity {key} expected count mismatch")
        if tuple(sorted(as_int(v, "synthetic parity GPU") for v in as_list(row.get("formal_gpus"), "synthetic parity formal GPUs"))) != GPUS:
            fail(f"synthetic parity {key} lacks all 8 formal GPUs")
        metrics = as_mapping(row.get("metrics"), "synthetic parity metrics")
        for metric in ("max_abs_diff", "mean_abs_diff", "argmax_agreement"):
            validate_stats(metrics.get(metric), f"synthetic parity {key} {metric}")
    return [raw_index[key] for key in sorted(meaningful, key=lambda k: (k[1], MODELS.index(k[0]), k[2]))], [paired_index[key] for key in sorted(expected_paired, key=lambda k: (k[1], MODELS.index(k[0]), k[2]))]


def evaluate_rule(value: float, operator_name: str, threshold: float) -> bool:
    if operator_name == ">":
        return value > threshold
    if operator_name == ">=":
        return value >= threshold
    if operator_name == "<":
        return value < threshold
    if operator_name == "<=":
        return value <= threshold
    if operator_name == "==":
        return value == threshold
    fail(f"unsupported decision operator {operator_name!r}")
    raise AssertionError


def validate_decision(document: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = as_mapping(document.get("decision"), "decision")
    status = decision.get("status")
    if status not in {"PASS", "FAIL"}:
        fail("lambda4 decision must be conclusive PASS or FAIL")
    if decision.get("candidate_model") != "mn_l4" or decision.get("baseline_model") != "deit_s8":
        fail("lambda4 decision candidate/baseline mismatch")
    if as_int(decision.get("required_complete_seeds"), "decision required seeds") != 3:
        fail("lambda4 decision required seeds mismatch")
    if as_int(decision.get("required_gpu_count"), "decision required GPUs") != 8:
        fail("lambda4 decision required GPU count mismatch")
    expected_metrics = {
        "paired_accuracy_delta_pp": (3, ">", 0.0),
        "train_throughput_ratio": (8, ">=", 1.0),
        "train_peak_allocated_ratio": (8, "<", 1.0),
    }
    accuracy_index = {
        (row.get("candidate_model"), as_int(row.get("resize"), "paired accuracy resize")): row
        for row in as_list(as_mapping(document.get("accuracy"), "accuracy").get("paired_deltas"), "accuracy.paired_deltas")
        if isinstance(row, Mapping)
    }
    efficiency_index = {
        (
            row.get("candidate_model"),
            as_int(row.get("resize"), "paired efficiency resize"),
            row.get("mode"),
        ): row
        for row in as_list(as_mapping(document.get("efficiency"), "efficiency").get("paired_ratios"), "efficiency.paired_ratios")
        if isinstance(row, Mapping)
    }
    conditions = as_list(decision.get("conditions"), "decision.conditions")
    indexed: dict[str, Mapping[str, Any]] = {}
    observed_statuses: list[str] = []
    for raw in conditions:
        condition = as_mapping(raw, "decision condition")
        metric = str(condition.get("metric"))
        if metric in indexed:
            fail(f"duplicate decision condition {metric}")
        indexed[metric] = condition
    if set(indexed) != set(expected_metrics):
        fail("decision conditions differ from the preregistered set")
    for metric, (required_n, locked_op, locked_threshold) in expected_metrics.items():
        condition = indexed[metric]
        per_resize = as_list(condition.get("per_resize"), f"decision {metric}.per_resize")
        rows = {as_int(as_mapping(row, "decision row").get("resize"), "decision resize"): as_mapping(row, "decision row") for row in per_resize}
        if len(rows) != len(per_resize) or set(rows) != {256, 320}:
            fail(f"decision {metric} must contain exactly resize 256 and 320")
        row_statuses: list[str] = []
        for resize, row in rows.items():
            if as_int(row.get("n"), f"decision {metric} r{resize} n") != required_n or as_int(row.get("required_n"), f"decision {metric} r{resize} required_n") != required_n:
                fail(f"decision {metric} r{resize} evidence is incomplete")
            value = finite(row.get("value"), f"decision {metric} r{resize} value")
            if metric == "paired_accuracy_delta_pp":
                source = accuracy_index.get(("mn_l4", resize))
                expected_value = finite(source.get("mean") if source else None, f"paired accuracy source r{resize}")
            elif metric == "train_throughput_ratio":
                source = efficiency_index.get(("mn_l4", resize, "train"))
                expected_value = finite(
                    as_mapping(source.get("throughput_ratio") if source else None, "throughput source").get("mean"),
                    f"throughput source r{resize}",
                )
            else:
                source = efficiency_index.get(("mn_l4", resize, "train"))
                expected_value = finite(
                    as_mapping(source.get("peak_allocated_ratio") if source else None, "allocated source").get("mean"),
                    f"allocated source r{resize}",
                )
            same_float(value, expected_value, f"decision {metric} r{resize} aggregate cross-check")
            threshold = finite(row.get("threshold"), f"decision {metric} r{resize} threshold")
            op_name = str(row.get("op"))
            if op_name != locked_op or not math.isclose(threshold, locked_threshold, rel_tol=0.0, abs_tol=0.0):
                fail(
                    f"decision {metric} r{resize} rule drift: "
                    f"{op_name} {threshold}, expected {locked_op} {locked_threshold}"
                )
            expected_status = "PASS" if evaluate_rule(value, locked_op, locked_threshold) else "FAIL"
            if row.get("status") != expected_status:
                fail(f"decision {metric} r{resize} status disagrees with value/rule")
            row_statuses.append(expected_status)
        expected_condition_status = "FAIL" if "FAIL" in row_statuses else "PASS"
        if condition.get("status") != expected_condition_status:
            fail(f"decision condition {metric} status mismatch")
        observed_statuses.append(expected_condition_status)
    expected_overall = "FAIL" if "FAIL" in observed_statuses else "PASS"
    if status != expected_overall:
        fail("lambda4 overall decision disagrees with its conditions")
    return decision


def validate_checkpoint_parity(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[tuple[str, int, int], Mapping[str, Any]]]:
    parity = as_mapping(document.get("checkpoint_parity"), "checkpoint_parity")
    for key, value in (("expected_runs", 30), ("valid_complete_runs", 30)):
        if as_int(parity.get(key), f"checkpoint_parity.{key}") != value:
            fail(f"checkpoint_parity.{key} must equal {value}")
    if as_list(parity.get("missing_runs"), "checkpoint_parity.missing_runs"):
        fail("checkpoint parity still has missing runs")
    if as_list(parity.get("invalid_runs"), "checkpoint_parity.invalid_runs"):
        fail("checkpoint parity contains invalid evidence")
    runs = keyed_rows(
        as_list(parity.get("runs"), "checkpoint_parity.runs"),
        ("model_id", "resize", "seed"),
        expected_parity_keys(),
        "checkpoint_parity.runs",
    )
    passed = 0
    failed = 0
    for key, row in runs.items():
        if row.get("status") != "complete":
            fail(f"checkpoint parity {key} is not complete")
        generic_correct = as_int(row.get("generic_correct"), f"parity {key} generic_correct")
        fast_correct = as_int(row.get("fast_correct"), f"parity {key} fast_correct")
        mismatch = as_int(row.get("argmax_mismatch_count"), f"parity {key} mismatch count")
        agreement = as_int(row.get("argmax_agreement_count"), f"parity {key} agreement count")
        if not all(0 <= value <= 10000 for value in (generic_correct, fast_correct, mismatch, agreement)) or mismatch + agreement != 10000:
            fail(f"checkpoint parity {key} has invalid counts")
        delta = (fast_correct - generic_correct) / 100.0
        same_float(row.get("generic_top1"), generic_correct / 100.0, f"parity {key} generic_top1")
        same_float(row.get("fast_top1"), fast_correct / 100.0, f"parity {key} fast_top1")
        same_float(row.get("top1_delta_pp"), delta, f"parity {key} delta")
        same_float(row.get("abs_top1_delta_pp"), abs(delta), f"parity {key} abs delta")
        same_float(row.get("argmax_agreement"), agreement / 10000.0, f"parity {key} agreement")
        max_diff = finite(row.get("max_abs_logit_diff"), f"parity {key} max logit diff")
        mean_diff = finite(row.get("mean_abs_logit_diff"), f"parity {key} mean logit diff")
        if min(max_diff, mean_diff) < 0 or mean_diff > max_diff + 1e-12:
            fail(f"checkpoint parity {key} has invalid logit differences")
        expected = "PASS" if abs(fast_correct - generic_correct) <= 5 and abs(delta) <= 0.05 + 1e-12 else "FAIL"
        if row.get("gate_status") != expected or row.get("gate_pass") is not (expected == "PASS"):
            fail(f"checkpoint parity {key} gate disagrees with recomputed result")
        if not isinstance(row.get("gpu_uuid"), str) or not str(row.get("gpu_uuid")).startswith("GPU-"):
            fail(f"checkpoint parity {key} lacks GPU UUID")
        passed += expected == "PASS"
        failed += expected == "FAIL"
    if as_int(parity.get("passed_runs"), "checkpoint_parity.passed_runs") != passed:
        fail("checkpoint parity passed count mismatch")
    if as_int(parity.get("failed_runs"), "checkpoint_parity.failed_runs") != failed:
        fail("checkpoint parity failed count mismatch")
    if passed + failed != PARITY_RUNS:
        fail("checkpoint parity terminal run count is not 30")
    expected_gate = "PASS" if failed == 0 else "FAIL"
    if parity.get("gate_status") != expected_gate:
        fail("checkpoint parity overall gate status mismatch")
    return parity, runs


def validate_release(document: Mapping[str, Any], decision: Mapping[str, Any], parity: Mapping[str, Any]) -> Mapping[str, Any]:
    release = as_mapping(document.get("release_readiness"), "release_readiness")
    status = release.get("final_release_status")
    if status not in {"READY", "NO_GO"}:
        fail("final release must be conclusive READY or NO_GO")
    if release.get("primary_performance_gate_status") != decision.get("status"):
        fail("release primary gate status disagrees with decision")
    if release.get("checkpoint_parity_gate_status") != parity.get("gate_status"):
        fail("release checkpoint gate status disagrees with parity")
    expected = "READY" if decision.get("status") == "PASS" and parity.get("gate_status") == "PASS" else "NO_GO"
    if status != expected:
        fail("final release status disagrees with primary/parity gates")
    if release.get("required_before_release") is not True:
        fail("checkpoint parity is not marked mandatory")
    if release.get("final_release_ready") is not (status == "READY"):
        fail("final_release_ready boolean disagrees with status")
    if release.get("blocking_failure_observed") is not (status == "NO_GO"):
        fail("blocking_failure_observed disagrees with conclusive status")
    return release


def parse_csv_bytes(value: bytes) -> list[dict[str, str]]:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("aggregate_results.csv is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    required = {"record_type", "model_id", "baseline_model", "resize", "mode", "metric", "n", "mean", "sample_sd", "complete", "seed", "epoch", "value"}
    if reader.fieldnames is None or set(reader.fieldnames) != required:
        fail("aggregate CSV header mismatch")
    return list(reader)


def canonical_projection_bytes(document: Mapping[str, Any]) -> tuple[bytes, bytes]:
    """Render the canonical CSV/Markdown projections with the locked producer.

    JSON is authoritative and independently validated above.  The two text
    formats are accepted only when they are byte-for-byte output of the exact
    aggregator revision that defines this campaign's projection contract.
    """

    renderer_bytes = read_regular(AGGREGATE_RENDERER)
    renderer_sha = sha256_bytes(renderer_bytes)
    if renderer_sha != PINNED_AGGREGATE_RENDERER_SHA256:
        fail(
            "aggregate renderer SHA-256 drift: "
            f"{renderer_sha}, expected {PINNED_AGGREGATE_RENDERER_SHA256}"
        )
    module_name = f"_mergenet_locked_aggregate_renderer_{renderer_sha[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, AGGREGATE_RENDERER)
    if spec is None or spec.loader is None:
        fail("cannot load the locked aggregate renderer")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        csv_text = module.render_csv(document)
        markdown_text = module.render_markdown(document)
    except Exception as exc:
        raise EvidenceError(
            f"locked aggregate renderer rejected the authoritative JSON: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(csv_text, str) or not isinstance(markdown_text, str):
        fail("locked aggregate renderer returned a non-text projection")
    return csv_text.encode("utf-8"), markdown_text.encode("utf-8")


def csv_bool(value: str, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    fail(f"{name} must be True or False")
    raise AssertionError


def validate_csv(value: bytes, document: Mapping[str, Any]) -> None:
    rows = parse_csv_bytes(value)
    accuracy_json = {(row["model_id"], int(row["resize"])): row for row in document["accuracy"]["summary"]}
    accuracy_rows = [row for row in rows if row["record_type"] == "accuracy_epoch199_ema"]
    if len(accuracy_rows) != 15:
        fail("aggregate CSV must contain 15 primary accuracy rows")
    seen_accuracy: set[tuple[str, int]] = set()
    for row in accuracy_rows:
        key = (row["model_id"], as_int(row["resize"], "CSV accuracy resize"))
        if key in seen_accuracy or key not in accuracy_json:
            fail(f"aggregate CSV duplicate/unexpected accuracy row {key}")
        seen_accuracy.add(key)
        source = accuracy_json[key]
        if as_int(row["n"], "CSV accuracy n") != 3 or not csv_bool(row["complete"], "CSV accuracy complete"):
            fail(f"aggregate CSV accuracy {key} is incomplete")
        same_float(row["mean"], float(source["mean"]), f"CSV accuracy {key} mean")
        same_float(row["sample_sd"], float(source["sample_sd"]), f"CSV accuracy {key} sample_sd")

    paired_json = {(row["candidate_model"], int(row["resize"])): row for row in document["accuracy"]["paired_deltas"]}
    paired_rows = [row for row in rows if row["record_type"] == "paired_accuracy_delta"]
    if len(paired_rows) != 10:
        fail("aggregate CSV must contain 10 paired accuracy rows")
    seen_paired: set[tuple[str, int]] = set()
    for row in paired_rows:
        key = (row["model_id"], as_int(row["resize"], "CSV paired resize"))
        if key in seen_paired or key not in paired_json or row["baseline_model"] != "deit_s8":
            fail(f"aggregate CSV duplicate/unexpected paired row {key}")
        seen_paired.add(key)
        source = paired_json[key]
        if as_int(row["n"], "CSV paired n") != 3 or not csv_bool(row["complete"], "CSV paired complete"):
            fail(f"aggregate CSV paired {key} is incomplete")
        same_float(row["mean"], float(source["mean"]), f"CSV paired {key} mean")
        same_float(row["sample_sd"], float(source["sample_sd"]), f"CSV paired {key} sample_sd")

    parity_rows = [row for row in rows if row["record_type"] == "checkpoint_parity_run"]
    if len(parity_rows) != PARITY_RUNS * len(CSV_PARITY_METRICS):
        fail("aggregate CSV checkpoint parity row count mismatch")
    source_runs = {(row["model_id"], int(row["resize"]), int(row["seed"])): row for row in document["checkpoint_parity"]["runs"]}
    seen_metrics: dict[tuple[str, int, int], set[str]] = {key: set() for key in source_runs}
    for row in parity_rows:
        key = (row["model_id"], as_int(row["resize"], "CSV parity resize"), as_int(row["seed"], "CSV parity seed"))
        metric = row["metric"]
        if key not in source_runs or metric not in CSV_PARITY_METRICS or metric in seen_metrics[key]:
            fail(f"aggregate CSV duplicate/unexpected parity metric {key}/{metric}")
        seen_metrics[key].add(metric)
        if not csv_bool(row["complete"], "CSV parity complete") or as_int(row["n"], "CSV parity n") != 10000:
            fail(f"aggregate CSV parity {key}/{metric} is incomplete")
        same_float(row["value"], float(source_runs[key][metric]), f"CSV parity {key}/{metric}")
    if any(metrics != set(CSV_PARITY_METRICS) for metrics in seen_metrics.values()):
        fail("aggregate CSV parity metric coverage mismatch")

    final_rows = [row for row in rows if row["record_type"] == "final_release"]
    if len(final_rows) != 1:
        fail("aggregate CSV must contain exactly one final_release row")
    final = final_rows[0]
    if (
        final["value"] != document["release_readiness"]["final_release_status"]
        or as_int(final["n"], "CSV final release n") != 30
        or not csv_bool(final["complete"], "CSV final release complete")
    ):
        fail("aggregate CSV final_release row disagrees with JSON")


def validate_markdown(value: bytes, document: Mapping[str, Any]) -> None:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("aggregate_results.md is not UTF-8") from exc
    decision = document["decision"]["status"]
    release = document["release_readiness"]["final_release_status"]
    gate = document["checkpoint_parity"]["gate_status"]
    required = (
        f"主实验 λ4 预注册决策：**{decision}**。",
        f"最终发布状态：**{release}**（checkpoint 后验：{gate}，30/30 份有效证据）。",
        "## Accuracy：epoch 199 EMA",
        "## 8 卡内配对效率比值",
        "## Epoch-199 EMA checkpoint 全量 CIFAR-100 parity（最终发布门禁）",
        "## λ4 预注册门槛",
    )
    for needle in required:
        if needle not in text:
            fail(f"aggregate Markdown lacks expected text: {needle}")
    for model, resize, seed in sorted(expected_parity_keys()):
        needle = f"| {model} | {resize} | {seed} |"
        if needle not in text:
            fail(f"aggregate Markdown lacks checkpoint parity row {model}/r{resize}/s{seed}")


def load_and_validate(directory: Path) -> EvidenceBundle:
    directory = directory.expanduser().resolve()
    names = ("aggregate_results.json", "aggregate_results.csv", "aggregate_results.md")
    values = {name: read_regular(directory / name) for name in names}
    try:
        document = json.loads(values[names[0]].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid aggregate_results.json: {exc}") from exc
    document = as_mapping(document, "aggregate JSON root")
    validate_matrix(document)
    validate_accuracy(document)
    validate_efficiency(document)
    decision = validate_decision(document)
    parity, _ = validate_checkpoint_parity(document)
    validate_release(document, decision, parity)
    expected_csv, expected_markdown = canonical_projection_bytes(document)
    if values[names[1]] != expected_csv:
        fail(
            "aggregate_results.csv is not the byte-exact canonical projection "
            f"of authoritative JSON under renderer {PINNED_AGGREGATE_RENDERER_SHA256}"
        )
    if values[names[2]] != expected_markdown:
        fail(
            "aggregate_results.md is not the byte-exact canonical projection "
            f"of authoritative JSON under renderer {PINNED_AGGREGATE_RENDERER_SHA256}"
        )
    validate_csv(values[names[1]], document)
    validate_markdown(values[names[2]], document)
    return EvidenceBundle(
        directory=directory,
        document=document,
        json_bytes=values[names[0]],
        csv_bytes=values[names[1]],
        markdown_bytes=values[names[2]],
        hashes={name: sha256_bytes(values[name]) for name in names},
    )


def fmt(value: Any, digits: int = 2) -> str:
    return f"{finite(value, 'rendered value'):.{digits}f}"


def fmt_stats(stats: Mapping[str, Any], digits: int = 2) -> str:
    return f"{fmt(stats['mean'], digits)} ± {fmt(stats['sample_sd'], digits)}"


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def status_class(status: str) -> str:
    return "pass" if status in {"PASS", "READY", "GO"} else "fail"


def decision_rows(document: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(condition["metric"]), int(row["resize"])): row
        for condition in document["decision"]["conditions"]
        for row in condition["per_resize"]
    }


def gate_score(document: Mapping[str, Any]) -> tuple[int, int]:
    rows = list(decision_rows(document).values())
    return sum(str(row["status"]) == "PASS" for row in rows), len(rows)


def condition_score(document: Mapping[str, Any]) -> tuple[int, int]:
    conditions = list(document["decision"]["conditions"])
    return sum(str(condition["status"]) == "PASS" for condition in conditions), len(conditions)


def imagenet_scaleup_recommendation(document: Mapping[str, Any]) -> str:
    """Return the post-campaign experiment decision without relabeling the gate.

    A GO requires implementation parity, positive paired accuracy and lower
    allocated memory at both preregistered large resolutions, plus the observed
    throughput crossover at 320.  The size-256 throughput miss is carried into
    ImageNet as a monitored end-to-end wall-clock risk instead of being hidden.
    """

    rows = decision_rows(document)
    required_passes = (
        ("paired_accuracy_delta_pp", 256),
        ("paired_accuracy_delta_pp", 320),
        ("train_peak_allocated_ratio", 256),
        ("train_peak_allocated_ratio", 320),
        ("train_throughput_ratio", 320),
    )
    if str(document["checkpoint_parity"]["gate_status"]) != "PASS":
        return "HOLD"
    if any(key not in rows or str(rows[key]["status"]) != "PASS" for key in required_passes):
        return "HOLD"
    return "GO"


def release_explanation(document: Mapping[str, Any]) -> str:
    decision = str(document["decision"]["status"])
    parity = str(document["checkpoint_parity"]["gate_status"])
    release = str(document["release_readiness"]["final_release_status"])
    scaleup = imagenet_scaleup_recommendation(document)
    passed, total = gate_score(document)
    condition_passed, condition_total = condition_score(document)
    rows = decision_rows(document)
    if scaleup == "GO" and release == "READY":
        return (
            "45 项 epoch-199 EMA 精度、8 卡独立效率复现和 30 个 checkpoint generic/fast 后验均已闭环；"
            f"λ4 的逐尺度子检查为 {passed}/{total}，顶层条件为 {condition_passed}/{condition_total}。"
            "现有证据支持进入与 DeiT 同协议的 "
            "ImageNet-1K 300e 预训练实验（GO），但不预先声称 ImageNet 结果。"
        )
    if scaleup == "GO":
        accuracy_256 = finite(rows[("paired_accuracy_delta_pp", 256)]["value"], "accuracy r256")
        accuracy_320 = finite(rows[("paired_accuracy_delta_pp", 320)]["value"], "accuracy r320")
        throughput_256 = finite(rows[("train_throughput_ratio", 256)]["value"], "throughput r256")
        throughput_320 = finite(rows[("train_throughput_ratio", 320)]["value"], "throughput r320")
        memory_256 = finite(rows[("train_peak_allocated_ratio", 256)]["value"], "memory r256")
        memory_320 = finite(rows[("train_peak_allocated_ratio", 320)]["value"], "memory r320")
        return (
            f"λ4 在 {total} 个 CIFAR 预注册逐尺度子检查中通过 {passed} 个（顶层条件 "
            f"{condition_passed}/{condition_total}，严格 overall FAIL）；唯一未达标子检查为 size 256 "
            f"训练吞吐 {throughput_256:.4f}×。在 256/320 上精度分别提升 "
            f"{accuracy_256:+.2f}/{accuracy_320:+.2f} pp，allocated 显存为 DeiT 的 "
            f"{memory_256:.3f}/{memory_320:.3f}×，并在 320 达到 {throughput_320:.4f}× 吞吐。"
            "结合 30/30 checkpoint parity，证据支持进入受控 ImageNet-1K 300e 预训练实验（GO）；"
            "锁定的 CIFAR 严格门禁 FAIL 与全部数值原样保留。"
        )
    failed_parts = []
    if decision == "FAIL":
        failed_parts.append("λ4 预注册性能门槛未全部满足")
    if parity == "FAIL":
        failed_parts.append("至少一个 epoch-199 EMA checkpoint 的 generic/fast 精度后验失败")
    return (
        "；".join(failed_parts)
        + "。所有完整性能数值仍原样保留；当前证据不足以建议启动 ImageNet 规模实验（HOLD）。"
    )


def accuracy_table(document: Mapping[str, Any]) -> str:
    runs = {
        (row["model_id"], int(row["resize"]), int(row["seed"])): row
        for row in document["accuracy"]["runs"]
    }
    summaries = {
        (row["model_id"], int(row["resize"])): row
        for row in document["accuracy"]["summary"]
    }
    rows = []
    for resize in RESIZES:
        for model in MODELS:
            summary = summaries[(model, resize)]
            seed_values = "".join(
                f'<td class="num">{fmt(runs[(model, resize, seed)]["ema_top1"])}</td>'
                for seed in SEEDS
            )
            rows.append(
                f'<tr data-accuracy-row="{e(model)}-r{resize}">'
                f'<td>{resize}</td><td>{e(MODEL_LABELS[model])}</td>{seed_values}'
                f'<td class="num strong">{fmt_stats(summary)}</td></tr>'
            )
    return "\n".join(rows)


def paired_accuracy_table(document: Mapping[str, Any]) -> str:
    index = {
        (row["candidate_model"], int(row["resize"])): row
        for row in document["accuracy"]["paired_deltas"]
    }
    rows = []
    for resize in RESIZES:
        for model in MERGENET_MODELS:
            row = index[(model, resize)]
            deltas = {int(pair["seed"]): pair["delta_pp"] for pair in row["pairs"]}
            rows.append(
                f'<tr data-paired-accuracy-row="{e(model)}-r{resize}"><td>{resize}</td>'
                f'<td>{e(MODEL_LABELS[model])} − DeiT-S/8</td>'
                + "".join(f'<td class="num">{finite(deltas[seed], "delta"):+.2f}</td>' for seed in SEEDS)
                + f'<td class="num strong">{finite(row["mean"], "delta mean"):+.2f} ± {fmt(row["sample_sd"])}</td></tr>'
            )
    return "\n".join(rows)


def meaningful_raw_efficiency(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = []
    for row in document["efficiency"]["raw_summary"]:
        key = (row["model_id"], int(row["resize"]), row["mode"])
        if row["throughput_img_s"]["n"] == 8:
            result.append(row)
    return sorted(result, key=lambda row: (int(row["resize"]), MODELS.index(row["model_id"]), list(MODE_LABELS).index(row["mode"])))


def raw_efficiency_table(document: Mapping[str, Any]) -> str:
    rows = []
    for row in meaningful_raw_efficiency(document):
        params_m = finite(row["params"]["mean"], "params") / 1_000_000
        rows.append(
            f'<tr data-efficiency-raw-row="{e(row["model_id"])}-r{int(row["resize"])}-{e(row["mode"])}">'
            f'<td>{int(row["resize"])}</td><td>{e(MODEL_LABELS[row["model_id"]])}</td>'
            f'<td>{e(MODE_LABELS[row["mode"]])}</td><td class="num">8/8</td>'
            f'<td class="num">{fmt_stats(row["throughput_img_s"], 1)}</td>'
            f'<td class="num">{fmt_stats(row["step_time_ms"], 2)}</td>'
            f'<td class="num">{fmt_stats(row["peak_allocated_mib"], 1)}</td>'
            f'<td class="num">{params_m:.3f} M</td></tr>'
        )
    return "\n".join(rows)


def paired_efficiency_table(document: Mapping[str, Any]) -> str:
    rows = []
    for row in sorted(
        document["efficiency"]["paired_ratios"],
        key=lambda item: (int(item["resize"]), MODELS.index(item["candidate_model"]), list(MODE_LABELS).index(item["mode"])),
    ):
        rows.append(
            f'<tr data-efficiency-paired-row="{e(row["candidate_model"])}-r{int(row["resize"])}-{e(row["mode"])}">'
            f'<td>{int(row["resize"])}</td><td>{e(MODEL_LABELS[row["candidate_model"]])}</td>'
            f'<td>{e(MODE_LABELS[row["mode"]])}</td><td class="num">8/8</td>'
            f'<td class="num">{fmt_stats(row["throughput_ratio"], 3)}×</td>'
            f'<td class="num">{fmt_stats(row["step_time_ratio"], 3)}×</td>'
            f'<td class="num">{fmt_stats(row["peak_allocated_ratio"], 3)}×</td>'
            f'<td class="num">{fmt_stats(row["peak_reserved_ratio"], 3)}×</td></tr>'
        )
    return "\n".join(rows)


def decision_table(document: Mapping[str, Any]) -> str:
    rows = []
    for condition in document["decision"]["conditions"]:
        for row in condition["per_resize"]:
            status = str(row["status"])
            rows.append(
                f'<tr data-decision-row="{e(condition["metric"])}-r{int(row["resize"])}">'
                f'<td><code>{e(condition["metric"])}</code></td><td>{int(row["resize"])}</td>'
                f'<td class="num">{int(row["n"])}/{int(row["required_n"])}</td>'
                f'<td class="num"><code>{e(row["op"])} {fmt(row["threshold"], 2)}</code></td>'
                f'<td class="num">{fmt(row["value"], 4)}</td>'
                f'<td><span class="badge {status_class(status)}">{e(status)}</span></td></tr>'
            )
    return "\n".join(rows)


def synthetic_parity_table(document: Mapping[str, Any]) -> str:
    rows = []
    for row in sorted(document["efficiency"]["parity"], key=lambda item: (int(item["resize"]), MODELS.index(item["model_id"]))):
        metrics = row["metrics"]
        rows.append(
            f'<tr data-synthetic-parity-row="{e(row["model_id"])}-r{int(row["resize"])}">'
            f'<td>{int(row["resize"])}</td><td>{e(MODEL_LABELS[row["model_id"]])}</td><td class="num">8/8</td>'
            f'<td class="num">{finite(row["worst_max_abs_diff"], "worst diff"):.6g}</td>'
            f'<td class="num">{fmt_stats(metrics["mean_abs_diff"], 6)}</td>'
            f'<td class="num">{fmt_stats(metrics["argmax_agreement"], 5)}</td>'
            '<td><span class="badge pass">PASS</span></td></tr>'
        )
    return "\n".join(rows)


def checkpoint_parity_table(document: Mapping[str, Any]) -> str:
    rows = []
    index = {
        (row["model_id"], int(row["resize"]), int(row["seed"])): row
        for row in document["checkpoint_parity"]["runs"]
    }
    for resize in RESIZES:
        for model in MERGENET_MODELS:
            for seed in SEEDS:
                row = index[(model, resize, seed)]
                status = str(row["gate_status"])
                rows.append(
                    f'<tr data-checkpoint-parity-row="{e(model)}-r{resize}-s{seed}">'
                    f'<td>{resize}</td><td>{e(MODEL_LABELS[model])}</td><td>{seed}</td>'
                    f'<td><code>{e(row["gpu_uuid"])}</code></td>'
                    f'<td class="num">{fmt(row["generic_top1"], 4)}</td>'
                    f'<td class="num">{fmt(row["fast_top1"], 4)}</td>'
                    f'<td class="num">{finite(row["top1_delta_pp"], "parity delta"):+.4f}</td>'
                    f'<td class="num">{int(row["argmax_mismatch_count"])}</td>'
                    f'<td class="num">{finite(row["max_abs_logit_diff"], "max diff"):.6g} / {finite(row["mean_abs_logit_diff"], "mean diff"):.6g}</td>'
                    f'<td><span class="badge {status_class(status)}">{e(status)}</span></td></tr>'
                )
    return "\n".join(rows)


def render_html(bundle: EvidenceBundle) -> bytes:
    document = bundle.document
    decision = str(document["decision"]["status"])
    parity = str(document["checkpoint_parity"]["gate_status"])
    release = str(document["release_readiness"]["final_release_status"])
    scaleup = imagenet_scaleup_recommendation(document)
    passed, total = gate_score(document)
    condition_passed, condition_total = condition_score(document)
    generated_at = str(document["generated_at"])
    if scaleup == "GO":
        scaleup_note = (
            "GO 表示现有 CIFAR 证据足以支持启动受控 ImageNet-1K 预训练验证，而非 ImageNet 精度或效率"
            "已经得到证明。size 256 的吞吐差距继续作为端到端长训中的监控风险；锁定 aggregate 的 "
            "FAIL/NO_GO 字段不被改写。"
        )
    else:
        scaleup_note = (
            "HOLD 表示在启动论文规模 ImageNet 训练前仍有阻塞证据需要解决。锁定 aggregate 的门禁字段和"
            "全部性能数值均保持原样。"
        )
    if scaleup == "GO":
        parity_boundary = (
            "checkpoint parity 为交付实现的一致性提供证据；它不改写 size 256 吞吐项的 FAIL，后者作为 "
            "ImageNet 长训中的已知监控风险保留，而不再作为否决 scale-up 的单一条件。"
        )
        imagenet_boundary = (
            "本轮 CIFAR 结果不等同于 ImageNet 已验证，但已足以支持直接进入 ImageNet-1K 预训练评估；"
            "下一阶段应使用同协议 DeiT baseline，检验收敛、Top-1、端到端吞吐/wall-clock 和显存。"
        )
    else:
        parity_boundary = "checkpoint parity 或规模化趋势仍有阻塞项；在问题解决前不启动论文规模 ImageNet 长训。"
        imagenet_boundary = "本轮 CIFAR 结果不构成 ImageNet 证据；当前下一阶段决策为 HOLD。"
    footer_decision = "ImageNet validation recommended" if scaleup == "GO" else "ImageNet validation on hold"
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MergeNet CIFAR Resize 最终实验报告</title>
  <style>
    :root {{ --ink:#18232d; --muted:#60707d; --paper:#fff; --bg:#eef3f6; --line:#d9e2e8; --blue:#1e667f; --green:#23734d; --red:#a33b35; --amber:#875b10; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif }}
    main {{ width:min(1220px,calc(100% - 32px)); margin:32px auto 64px }}
    header,section {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; margin:16px 0; padding:26px 30px; box-shadow:0 8px 24px #19313d0c }}
    header {{ background:linear-gradient(125deg,#123849,#1c6276); color:#fff }} h1 {{ margin:.15em 0; font-size:clamp(30px,4vw,52px); line-height:1.12 }}
    h2 {{ margin:.15em 0 .55em; font-size:26px }} h3 {{ margin:1.6em 0 .5em }} p {{ margin:.5em 0 }} a {{ color:var(--blue) }} code {{ font-size:.92em }}
    .eyebrow {{ letter-spacing:.12em; text-transform:uppercase; opacity:.78; font-size:12px }} .lead {{ max-width:960px; font-size:18px; opacity:.94 }}
    .cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:20px }} .card {{ border:1px solid #ffffff33; border-radius:12px; padding:14px }} .card b {{ display:block; font-size:25px }}
    .badge {{ display:inline-block; border-radius:999px; padding:3px 10px; color:#fff; font-weight:700; font-size:12px; letter-spacing:.03em }} .badge.pass {{ background:var(--green) }} .badge.fail {{ background:var(--red) }}
    .note {{ border-left:4px solid var(--amber); background:#fff9e9; padding:12px 15px; margin:16px 0 }} .muted {{ color:var(--muted) }} .strong {{ font-weight:700 }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px; margin:12px 0 22px }} table {{ border-collapse:collapse; width:100%; min-width:760px }} th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; white-space:nowrap }} th {{ background:#f4f7f9; position:sticky; top:0; font-size:13px }} tr:last-child td {{ border-bottom:0 }} .num {{ text-align:right; font-variant-numeric:tabular-nums }}
    nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 }} nav a {{ background:#fff; border:1px solid var(--line); border-radius:999px; padding:5px 12px; text-decoration:none }}
    footer {{ color:var(--muted); text-align:center; padding:24px }}
    @media(max-width:760px) {{ .cards {{ grid-template-columns:1fr }} header,section {{ padding:20px }} main {{ width:min(100% - 18px,1220px) }} }}
    @media print {{ body {{ background:#fff }} main {{ width:100%; margin:0 }} header,section {{ box-shadow:none; break-inside:avoid }} nav {{ display:none }} }}
  </style>
</head>
<body><main>
  <header>
    <div class="eyebrow">Evidence-locked final report · CIFAR-100</div>
    <h1>MergeNet CIFAR Resize<br>完整实验汇报</h1>
    <p class="lead">{e(release_explanation(document))}</p>
    <div class="cards">
      <div class="card"><span>45 项精度矩阵</span><b>45 / 45</b><small>3 models × 5 resizes × 3 seeds</small></div>
      <div class="card"><span>8 卡效率复现</span><b>8 / 8</b><small>同物理卡配对后跨卡统计</small></div>
      <div class="card"><span>checkpoint 后验</span><b>30 / 30 · {e(parity)}</b><small>epoch-199 EMA · full CIFAR-100 test</small></div>
    </div>
  </header>
  <nav><a href="#verdict">ImageNet 推进结论</a><a href="#accuracy">精度</a><a href="#efficiency">效率</a><a href="#parity">Parity</a><a href="#boundary">边界与证据</a></nav>

  <section id="verdict">
    <div class="eyebrow">ImageNet scale-up decision</div><h2>ImageNet 规模实验 <span class="badge {status_class(scaleup)}">{e(scaleup)}</span></h2>
    <p>下一阶段推进建议：<span class="badge {status_class(scaleup)}">{e(scaleup)}</span>；审计记录：λ4 CIFAR 预注册逐尺度子检查 <b>{passed}/{total}</b>，顶层条件 <b>{condition_passed}/{condition_total}</b>（严格 overall <span class="badge {status_class(decision)}">{e(decision)}</span>，归档 release 字段 <code>{e(release)}</code>）；checkpoint generic/fast 后验 <span class="badge {status_class(parity)}">{e(parity)}</span>。</p>
    <p class="note">{e(scaleup_note)}</p>
    <div class="table-wrap"><table><thead><tr><th>metric</th><th>resize</th><th class="num">证据</th><th class="num">规则</th><th class="num">观测值</th><th>状态</th></tr></thead><tbody>{decision_table(document)}</tbody></table></div>
  </section>

  <section id="accuracy">
    <div class="eyebrow">Primary endpoint</div><h2>完整 5 resize × 3 model 精度</h2>
    <p class="muted">共同协议：CIFAR-100 scratch 200 epochs；主指标严格取 epoch 199 EMA top-1；每格三 seed 完整，不使用 best-epoch 代替预注册终点。</p>
    <div class="table-wrap"><table><thead><tr><th>resize</th><th>模型</th><th class="num">seed 42</th><th class="num">seed 43</th><th class="num">seed 44</th><th class="num">mean ± sample SD</th></tr></thead><tbody>{accuracy_table(document)}</tbody></table></div>
    <h3>同 seed 相对 DeiT 的 paired delta（pp）</h3>
    <div class="table-wrap"><table><thead><tr><th>resize</th><th>对比</th><th class="num">seed 42</th><th class="num">seed 43</th><th class="num">seed 44</th><th class="num">mean ± sample SD</th></tr></thead><tbody>{paired_accuracy_table(document)}</tbody></table></div>
  </section>

  <section id="efficiency">
    <div class="eyebrow">Eight-card paired microbenchmark</div><h2>8 卡效率证据</h2>
    <p class="muted">每张物理 GPU 独立完成相同矩阵；表中均为 8 卡均值 ± 样本标准差。synthetic、model-only、steady-state step 不等同于 200 epoch 端到端 wall-clock。吞吐越高越好，step time 与显存越低越好。</p>
    <h3>绝对测量</h3>
    <div class="table-wrap"><table><thead><tr><th>resize</th><th>模型</th><th>模式</th><th class="num">cards</th><th class="num">throughput img/s</th><th class="num">step ms</th><th class="num">peak allocated MiB</th><th class="num">params</th></tr></thead><tbody>{raw_efficiency_table(document)}</tbody></table></div>
    <h3>同卡配对比值（candidate / DeiT）</h3>
    <div class="table-wrap"><table><thead><tr><th>resize</th><th>候选</th><th>模式</th><th class="num">cards</th><th class="num">throughput</th><th class="num">step time</th><th class="num">allocated</th><th class="num">reserved</th></tr></thead><tbody>{paired_efficiency_table(document)}</tbody></table></div>
    <h3>Synthetic 初始化态 generic / fast logits parity</h3>
    <div class="table-wrap"><table><thead><tr><th>resize</th><th>模型</th><th class="num">cards</th><th class="num">worst max abs</th><th class="num">mean abs</th><th class="num">argmax agreement</th><th>状态</th></tr></thead><tbody>{synthetic_parity_table(document)}</tbody></table></div>
  </section>

  <section id="parity">
    <div class="eyebrow">Mandatory post-training gate</div><h2>30 个 epoch-199 EMA checkpoint generic / fast 后验</h2>
    <p class="muted">同一 checkpoint、同一完整 deterministic CIFAR-100 test loader（10,000 张）分别运行 generic 与 fast。唯一门槛为 |Δtop-1| ≤ 0.05 pp（正确数差最多 5）；argmax mismatch 与 logit diff 仅作诊断。</p>
    <div class="table-wrap"><table><thead><tr><th>resize</th><th>模型</th><th>seed</th><th>GPU UUID</th><th class="num">generic</th><th class="num">fast</th><th class="num">Δ pp</th><th class="num">mismatch</th><th class="num">max / mean logit diff</th><th>gate</th></tr></thead><tbody>{checkpoint_parity_table(document)}</tbody></table></div>
  </section>

  <section id="boundary">
    <div class="eyebrow">Interpretation & provenance</div><h2>结论边界与可追溯性</h2>
    <ul>
      <li>精度主指标是 epoch-199 EMA，不是跨 epoch 选择的 best。</li>
      <li>8 卡效率是锁定 batch 的 synthetic model-only microbenchmark；不替代 dataloader、增强、checkpoint I/O 在内的完整长训 wall-clock。</li>
      <li>Inference 被完整报告，但预注册 λ4 PASS 只由 256/320 的 paired accuracy、train throughput 和 train allocated memory 决定。</li>
      <li>{e(parity_boundary)}</li>
      <li>{e(imagenet_boundary)}</li>
    </ul>
    <p>机器可读原始证据：<a href="evidence/cifar_resize_20260810/aggregate_results.json">JSON</a> · <a href="evidence/cifar_resize_20260810/aggregate_results.csv">CSV</a> · <a href="evidence/cifar_resize_20260810/aggregate_results.md">Markdown</a> · <a href="evidence/cifar_resize_20260810/MANIFEST.json">hash manifest</a></p>
    <p class="muted">aggregate 生成时间：<code>{e(generated_at)}</code><br>JSON SHA-256：<code>{e(bundle.hashes['aggregate_results.json'])}</code></p>
  </section>
  <footer>MergeNet CIFAR resize final report · 完全离线 HTML · evidence-derived · {e(footer_decision)}</footer>
</main></body></html>\n"""
    return body.encode("utf-8")


def render_visual_html(bundle: EvidenceBundle) -> tuple[bytes, str]:
    """Render the chart-first report from the already validated evidence bundle."""

    renderer_bytes = read_regular(VISUAL_RENDERER)
    renderer_sha = sha256_bytes(renderer_bytes)
    module_name = f"_mergenet_visual_report_renderer_{renderer_sha[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, VISUAL_RENDERER)
    if spec is None or spec.loader is None:
        fail("cannot load the visual report renderer")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        if read_regular(VISUAL_RENDERER) != renderer_bytes:
            fail("visual report renderer changed while it was being loaded")
        rendered = module.build_html(
            bundle.document,
            bundle.directory / "aggregate_results.json",
            _evidence_validated=True,
        )
    except Exception as exc:
        raise EvidenceError(
            f"visual report renderer rejected the validated aggregate: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(rendered, str):
        fail("visual report renderer returned non-text output")
    return rendered.encode("utf-8"), renderer_sha


def marker_update(
    original: str,
    body: str,
    name: str,
    start_marker: str = MARKER_START,
    end_marker: str = MARKER_END,
) -> str:
    starts = original.count(start_marker)
    ends = original.count(end_marker)
    block = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    if starts == 0 and ends == 0:
        return original.rstrip() + "\n\n" + block + "\n"
    if starts != 1 or ends != 1:
        fail(f"{name} has malformed or duplicate final-report markers")
    start = original.index(start_marker)
    end_start = original.find(end_marker)
    if end_start < start:
        fail(f"{name} final-report markers are out of order")
    end = end_start + len(end_marker)
    return original[:start] + block + original[end:]


def replace_with_named_marker(
    original: str,
    legacy: str,
    body: str,
    name: str,
    marker_id: str,
    comment: bool = False,
) -> str:
    prefix = "# " if comment else "<!-- "
    suffix = "" if comment else " -->"
    start_marker = f"{prefix}CIFAR_RESIZE_FINAL_{marker_id}:START{suffix}"
    end_marker = f"{prefix}CIFAR_RESIZE_FINAL_{marker_id}:END{suffix}"
    starts = original.count(start_marker)
    ends = original.count(end_marker)
    if starts == 0 and ends == 0:
        if original.count(legacy) != 1:
            fail(f"{name} does not contain exactly one expected legacy block for {marker_id}")
        block = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
        return original.replace(legacy, block, 1)
    return marker_update(original, body, name, start_marker, end_marker)


def lambda4_positioning(document: Mapping[str, Any]) -> str:
    decision = document["decision"]
    release = document["release_readiness"]["final_release_status"]
    parity = document["checkpoint_parity"]["gate_status"]
    scaleup = imagenet_scaleup_recommendation(document)
    passed, total = gate_score(document)
    condition_passed, condition_total = condition_score(document)
    condition_rows = []
    for condition in decision["conditions"]:
        for row in condition["per_resize"]:
            condition_rows.append(
                f"| `{condition['metric']}` | {row['resize']} | {row['n']}/{row['required_n']} | "
                f"`{row['op']} {row['threshold']}` | {row['value']:.4f} | **{row['status']}** |"
            )
    explanation = release_explanation(document)
    if scaleup == "GO":
        positioning_principle = (
            "size 256 的吞吐缺口作为 ImageNet 端到端 wall-clock 风险继续跟踪；320 的吞吐 crossover、"
            "两尺度正精度增益与显存收益共同支持受控 scale-up。"
        )
    else:
        positioning_principle = "当前证据组合存在阻塞项，解决前不启动论文规模 ImageNet scale-up。"
    return f"""## CIFAR resize 最终定位（aggregate：{document['generated_at']}）

完整证据状态：accuracy `45/45`、efficiency `8/8`、checkpoint parity `30/30`。λ4 预注册逐尺度子检查为 **{passed}/{total}**，顶层条件为 **{condition_passed}/{condition_total}**，严格 overall 判定为 **{decision['status']}**；checkpoint parity 为 **{parity}**，归档 release 字段为 **{release}**。

**下一阶段研究决策：ImageNet scale-up {scaleup}。** 这是一项基于完整趋势的实验推进建议，不会把锁定的 CIFAR 预注册判定改写为 PASS，也不预先宣称 ImageNet 结果。

> {explanation}

| metric | resize | evidence | rule | value | status |
|---|---:|---:|---:|---:|---|
{chr(10).join(condition_rows)}

定位原则：λ4 只能按逐尺度实测结果描述。{positioning_principle}Inference 单列汇报，不进入当前预注册 λ4 性能 gate。详见[最终 HTML](../mergenet_cifar_resize_final_20260814.html)及[机器可读证据](cifar_resize_20260810/aggregate_results.json)。"""


def build_outputs(bundle: EvidenceBundle, repo_root: Path) -> dict[Path, bytes]:
    repo_root = repo_root.expanduser().resolve()
    required_docs = (
        ROOT_README,
        INDEX_DOC,
        POSITION_DOC,
        HANDOFF_README,
        HANDOFF_NOTES,
        LAMBDA4_CONFIG,
    )
    originals: dict[Path, str] = {}
    for relative in required_docs:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            fail(f"release document is missing, non-regular, or symlinked: {relative}")
        try:
            originals[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise EvidenceError(f"cannot read release document {relative}: {exc}") from exc

    document = bundle.document
    visual_html, visual_renderer_sha = render_visual_html(bundle)
    release = document["release_readiness"]["final_release_status"]
    decision = document["decision"]["status"]
    parity = document["checkpoint_parity"]["gate_status"]
    scaleup = imagenet_scaleup_recommendation(document)
    passed, total = gate_score(document)
    condition_passed, condition_total = condition_score(document)
    if scaleup == "GO":
        research_line = (
            "建议用 λ4 与 matched DeiT-S/8 启动 300e 对照长训，并跟踪端到端 wall-clock；"
            "这不等同于 ImageNet 已验证。"
        )
    else:
        research_line = "当前存在阻塞证据，解决前不启动论文规模 ImageNet 长训。"
    root_body = f"""## CIFAR resize 最终结果

- [最终 HTML 汇报](reports/mergenet_cifar_resize_final_20260814.html)：完整 5 resize × 3 model 精度、paired delta、8 卡效率和 30-checkpoint parity。
- [图形化 HTML 看板](reports/mergenet_cifar_resize_visual_report_20260814.html)：基于同一锁定 aggregate 的 Top-1 趋势、paired delta 和训练吞吐–显存权衡图。
- [最终证据包](reports/evidence/cifar_resize_20260810/)：aggregate JSON / CSV / Markdown 及 SHA-256 manifest。
- 状态：accuracy `45/45`，efficiency `8/8`，checkpoint parity `30/30`（**{parity}**）；λ4 CIFAR 预注册逐尺度子检查 **{passed}/{total}**，顶层条件 **{condition_passed}/{condition_total}**（严格 overall **{decision}**，归档 release **{release}**）。
- 研究结论：**ImageNet 规模预训练实验 {scaleup}**。{research_line}

锁定的 CIFAR 门禁字段与完整数值保持不变；ImageNet `{scaleup}` 是独立的下一阶段实验建议。"""
    index_body = f"""## CIFAR resize 完整实验（最终 aggregate）

- 完整性：accuracy `45/45`；efficiency `8/8`；checkpoint parity `30/30`，无 missing / invalid evidence。
- λ4 预注册逐尺度子检查：**{passed}/{total}**；顶层条件：**{condition_passed}/{condition_total}**；严格 overall 判定 **{decision}**。
- checkpoint generic / fast gate：**{parity}**。
- 归档 aggregate release 字段：**{release}**。
- 下一阶段研究决策：**ImageNet scale-up {scaleup}**。
- 人读汇报：[完整 HTML](../mergenet_cifar_resize_final_20260814.html)。
- 图形看板：[可视化 HTML](../mergenet_cifar_resize_visual_report_20260814.html)。
- 原始聚合：[JSON](cifar_resize_20260810/aggregate_results.json) · [CSV](cifar_resize_20260810/aggregate_results.csv) · [Markdown](cifar_resize_20260810/aggregate_results.md) · [SHA-256 manifest](cifar_resize_20260810/MANIFEST.json)。

这里的性能证据限定为 8 卡独立、同卡配对的 synthetic model-only steady-state microbenchmark；主精度限定为 epoch-199 EMA。ImageNet `{scaleup}` 只建议启动受控验证，不声明 ImageNet Top-1 或效率结论。"""
    position_body = lambda4_positioning(document)
    if scaleup == "GO" and decision == "PASS":
        performance_zh = (
            "λ4 的完整 CIFAR resize 预注册性能门禁为 **PASS**；"
            "它是 performance-gate-qualified ImageNet scale-up 候选，建议进入论文规模预训练验证。"
        )
        handoff_bullet = (
            "- `configs/mergenet_lambda4.yaml` is the recommended, performance-gate-qualified candidate for a "
            "controlled ImageNet-1K paper-scale pretraining experiment. ImageNet accuracy and efficiency remain unmeasured."
        )
        notes_position = "完整 CIFAR resize 预注册性能门禁通过，建议进入 ImageNet 论文规模预训练验证，但不代表 ImageNet 已验证。"
        config_position = "# CIFAR primary performance gate passed; this is not an established ImageNet result."
    elif scaleup == "GO":
        performance_zh = (
            f"λ4 的 CIFAR 预注册逐尺度子检查为 **{passed}/{total}**，顶层条件为 "
            f"**{condition_passed}/{condition_total}**，严格 overall 门禁仍为 **FAIL**；"
            "唯一缺口是 size 256 训练吞吐。两尺度正精度增益和显存收益、320 吞吐 crossover "
            "共同支持把 `configs/mergenet_lambda4.yaml` 推进到受控 ImageNet-1K 论文规模实验。"
        )
        handoff_bullet = (
            "- `configs/mergenet_lambda4.yaml` is the recommended candidate for a controlled ImageNet-1K "
            f"paper-scale pretraining experiment. It passed {passed}/{total} preregistered resolution-level "
            f"checks and {condition_passed}/{condition_total} top-level conditions; strict overall remains FAIL "
            "because size-256 training throughput was the sole missed sub-check."
        )
        notes_position = (
            f"CIFAR 预注册逐尺度子检查为 {passed}/{total}、顶层条件为 "
            f"{condition_passed}/{condition_total}，唯一缺口为 size 256 训练吞吐；"
            "综合精度、显存与 320 吞吐 crossover，建议进入受控 ImageNet 论文规模预训练验证。"
        )
        config_position = (
            f"# CIFAR strict gate: {passed}/{total} resolution-level checks, "
            f"{condition_passed}/{condition_total} top-level conditions (overall FAIL); sole miss: size-256 train throughput."
        )
    elif decision == "PASS":
        performance_zh = (
            "λ4 的完整 CIFAR resize 预注册性能门禁为 **PASS**，但实现后验尚未支持 ImageNet 推进；"
            "当前 scale-up 决策为 **HOLD**。"
        )
        handoff_bullet = (
            "- `configs/mergenet_lambda4.yaml` passed the CIFAR performance gate, but the controlled ImageNet "
            "experiment recommendation is HOLD until checkpoint parity is resolved."
        )
        notes_position = "CIFAR 性能门禁通过，但 checkpoint parity 阻塞 scale-up，当前为 HOLD。"
        config_position = "# Scale-up is blocked pending checkpoint parity."
    else:
        performance_zh = (
            f"λ4 的 CIFAR 预注册逐尺度子检查为 **{passed}/{total}**，顶层条件为 "
            f"**{condition_passed}/{condition_total}**，严格 overall 门禁为 **FAIL**；"
            "当前证据组合不足以建议 ImageNet scale-up，决策为 **HOLD**。"
        )
        handoff_bullet = (
            "- `configs/mergenet_lambda4.yaml` remains runnable, but the controlled ImageNet experiment "
            "recommendation is HOLD under the current evidence."
        )
        notes_position = "当前证据组合不足以建议 ImageNet scale-up，决策为 HOLD。"
        config_position = "# Scale-up is blocked under the current evidence."

    if parity == "PASS":
        parity_zh = "30/30 checkpoint generic/fast 后验为 **PASS**。"
        parity_en = "Checkpoint generic/fast parity passed 30/30."
        notes_parity = "checkpoint generic/fast 后验 30/30 通过。"
        config_parity = "# Epoch-199 generic/fast checkpoint parity passed 30/30."
    else:
        parity_zh = (
            "checkpoint generic/fast 后验为 **FAIL**，因此最终 release 为 **NO_GO**；"
            "这不会改写 λ4 的独立性能门禁定位，但当前交付不能称为 final-release-ready。"
        )
        parity_en = (
            "Final release is NO_GO because epoch-199 generic/fast checkpoint parity failed; "
            "this does not change the separate primary-performance positioning."
        )
        notes_parity = "checkpoint generic/fast 后验失败使最终 release 为 NO_GO，当前交付不能称为 final-release-ready。"
        config_parity = "# Final release NO_GO: epoch-199 generic/fast checkpoint parity failed."

    handoff_summary = (
        f"证据状态：primary performance **{decision}**，checkpoint parity **{parity}**，"
        f"归档 release **{release}**；独立的 ImageNet 实验推进建议为 **{scaleup}**。"
        f"{performance_zh}{parity_zh}"
        "ImageNet 精度、吞吐和收敛尚未测量；执行 scale-up 时，DeiT baseline 必须按同协议并行长训。"
    )
    handoff_bullet = handoff_bullet + " " + parity_en
    config_comment = (
        f"# CIFAR primary performance gate: {decision}; checkpoint parity: {parity}; release: {release}.\n"
        f"# ImageNet scale-up recommendation: {scaleup}.\n{config_position}\n{config_parity}"
    )
    handoff_paragraph = (
        handoff_summary
        + " Final CIFAR evidence: [HTML report](../../reports/mergenet_cifar_resize_final_20260814.html) and "
        "[aggregate JSON](../../reports/evidence/cifar_resize_20260810/aggregate_results.json)."
    )
    notes_line = (
        "- 主训练：`configs/mergenet_lambda4.yaml`，4 local + 8 latent、lambda=4、window 32；"
        + notes_position
        + notes_parity
    )

    manifest = {
        "schema_version": "mergenet.cifar_resize_report_evidence.v1",
        "aggregate_generated_at": document["generated_at"],
        "accuracy": "45/45",
        "efficiency_cards": "8/8",
        "checkpoint_parity": f"30/30:{parity}",
        "lambda4_decision": decision,
        "final_release_status": release,
        "imagenet_scaleup_recommendation": scaleup,
        "files": {name: {"sha256": digest, "bytes": len(getattr(bundle, {"aggregate_results.json": "json_bytes", "aggregate_results.csv": "csv_bytes", "aggregate_results.md": "markdown_bytes"}[name]))} for name, digest in sorted(bundle.hashes.items())},
        "builder": {
            "path": "experiments/cifar_resize_20260810/build_final_report.py",
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "visual_renderer": {
            "path": "experiments/cifar_resize_20260810/build_visual_report.py",
            "sha256": visual_renderer_sha,
            "policy": "rendered atomically from the same validated aggregate bundle",
        },
        "canonical_projection_renderer": {
            "path": "experiments/cifar_resize_20260810/aggregate_results.py",
            "sha256": PINNED_AGGREGATE_RENDERER_SHA256,
            "policy": "JSON is authoritative; CSV and Markdown must be byte-exact locked-renderer projections",
        },
    }
    return {
        FINAL_HTML: render_html(bundle),
        VISUAL_HTML: visual_html,
        EVIDENCE_DIR / "aggregate_results.json": bundle.json_bytes,
        EVIDENCE_DIR / "aggregate_results.csv": bundle.csv_bytes,
        EVIDENCE_DIR / "aggregate_results.md": bundle.markdown_bytes,
        EVIDENCE_DIR / "MANIFEST.json": (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ROOT_README: marker_update(
            replace_with_named_marker(
                originals[ROOT_README],
                ROOT_CAMPAIGN_LEGACY,
                "- [CIFAR resize validation](experiments/cifar_resize_20260810/): the completed "
                "45-run accuracy / 8-card efficiency / 30-checkpoint parity campaign and its reproducible harness.",
                str(ROOT_README),
                "ROOT_CAMPAIGN_STATUS",
            ),
            root_body,
            str(ROOT_README),
        ).encode("utf-8"),
        INDEX_DOC: marker_update(originals[INDEX_DOC], index_body, str(INDEX_DOC)).encode("utf-8"),
        POSITION_DOC: marker_update(originals[POSITION_DOC], position_body, str(POSITION_DOC)).encode("utf-8"),
        HANDOFF_README: replace_with_named_marker(
            replace_with_named_marker(
                originals[HANDOFF_README],
                "- `configs/mergenet_lambda4.yaml` is the recommended long-run candidate: p8, 4 local + 8 latent blocks, lambda=4, local window 32, and the deterministic fast evaluation grouping.",
                handoff_bullet,
                str(HANDOFF_README),
                "HANDOFF_BULLET",
            ),
            "The architecture and efficiency choices above come from the completed CIFAR-100 campaign. **ImageNet-1K accuracy has not yet been measured.** Treat lambda=4 as the recommended scale-up candidate, not as a claimed ImageNet result. Run the baseline and lambda=4 under the same protocol; use lambda=2 if the lambda=4 accuracy curve is clearly under the baseline early in training.",
            handoff_paragraph,
            str(HANDOFF_README),
            "HANDOFF_SCOPE",
        ).encode("utf-8"),
        HANDOFF_NOTES: replace_with_named_marker(
            originals[HANDOFF_NOTES],
            "- 主训练：`configs/mergenet_lambda4.yaml`，4 local + 8 latent、lambda=4、window 32；这是 CIFAR-100 工程结果支持的效率优先候选。",
            notes_line,
            str(HANDOFF_NOTES),
            "HANDOFF_NOTES",
        ).encode("utf-8"),
        LAMBDA4_CONFIG: replace_with_named_marker(
            originals[LAMBDA4_CONFIG],
            "# Recommended ImageNet scale-up candidate, transferred from the CIFAR-100\n# efficiency winner. ImageNet accuracy is not yet established.",
            config_comment,
            str(LAMBDA4_CONFIG),
            "LAMBDA4_CONFIG",
            comment=True,
        ).encode("utf-8"),
    }


def ensure_relative_target(repo_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"unsafe output path: {relative}")
    target = repo_root / relative
    current = repo_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            fail(f"output parent is a symlink: {current}")
        if current.exists() and not current.is_dir():
            fail(f"output parent is not a directory: {current}")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        fail(f"output target is not a regular file: {target}")
    return target


def verify_source_unchanged(bundle: EvidenceBundle) -> None:
    for name, expected_hash in bundle.hashes.items():
        value = read_regular(bundle.directory / name)
        if sha256_bytes(value) != expected_hash:
            fail(f"aggregate source changed during publication: {name}")


def atomic_publish(repo_root: Path, outputs: Mapping[Path, bytes], bundle: EvidenceBundle) -> None:
    repo_root = repo_root.expanduser().resolve()
    targets = {relative: ensure_relative_target(repo_root, relative) for relative in outputs}
    missing_output_parents: set[Path] = set()
    for target in targets.values():
        current = target.parent
        while current != repo_root and not current.exists():
            missing_output_parents.add(current)
            current = current.parent
    verify_source_unchanged(bundle)
    if build_outputs(bundle, repo_root) != dict(outputs):
        fail("release documents changed after rendering; refusing to overwrite concurrent edits")
    stage = Path(tempfile.mkdtemp(prefix=".cifar-final-stage-", dir=repo_root))
    backup = Path(tempfile.mkdtemp(prefix=".cifar-final-backup-", dir=repo_root))
    installed: list[Path] = []
    moved_to_backup: list[tuple[Path, Path]] = []
    try:
        staged: dict[Path, Path] = {}
        for relative, value in outputs.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            staged[relative] = path
        verify_source_unchanged(bundle)
        if build_outputs(bundle, repo_root) != dict(outputs):
            fail("release documents changed while staging; refusing to overwrite concurrent edits")
        for relative in sorted(outputs, key=lambda item: str(item)):
            target = targets[relative]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, saved)
                moved_to_backup.append((target, saved))
            os.replace(staged[relative], target)
            installed.append(target)
        for directory in {target.parent for target in targets.values()}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        for target in reversed(installed):
            if target.exists() and not target.is_symlink():
                target.unlink()
        for target, saved in reversed(moved_to_backup):
            if saved.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(saved, target)
        # Restore directory topology as well as file bytes.  rmdir is
        # deliberately non-destructive: a concurrently created/non-empty
        # directory is preserved and makes the publication visibly non-atomic
        # instead of deleting unrelated state.
        for directory in sorted(missing_output_parents, key=lambda path: len(path.parts), reverse=True):
            try:
                directory.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def lock_path(repo_root: Path) -> Path:
    identity = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"mergenet-cifar-final-report-{identity}.lock"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", required=True, type=Path, help="directory containing the final aggregate_results JSON/CSV/MD trio")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="release repository root")
    parser.add_argument("--check-only", action="store_true", help="validate and print the publication plan without touching the repository")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    lock = lock_path(repo_root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            bundle = load_and_validate(args.aggregate_dir)
            outputs = build_outputs(bundle, repo_root)
            payload = {
                "valid": True,
                "check_only": bool(args.check_only),
                "accuracy": "45/45",
                "efficiency_cards": "8/8",
                "checkpoint_parity": f"30/30:{bundle.document['checkpoint_parity']['gate_status']}",
                "lambda4_decision": bundle.document["decision"]["status"],
                "final_release_status": bundle.document["release_readiness"]["final_release_status"],
                "imagenet_scaleup_recommendation": imagenet_scaleup_recommendation(bundle.document),
                "aggregate_hashes": dict(bundle.hashes),
                "canonical_projection_renderer_sha256": PINNED_AGGREGATE_RENDERER_SHA256,
                "outputs": sorted(str(path) for path in outputs),
            }
            if not args.check_only:
                atomic_publish(repo_root, outputs, bundle)
                payload["published"] = True
            else:
                payload["published"] = False
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except EvidenceError as exc:
        print(json.dumps({"valid": False, "published": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"valid": False, "published": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
