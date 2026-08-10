#!/usr/bin/env python3
"""Aggregate partial or complete CIFAR resize accuracy/efficiency evidence.

Primary accuracy is the preregistered epoch-199 EMA result.  Per-run best values
are intentionally kept in an appendix and never substitute for epoch 199.  GPU
efficiency comparisons are paired within physical card before aggregation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import operator
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = SCRIPT_DIR / "protocol.json"
SCHEMA_VERSION = 2
CHECKPOINT_PARITY_SCHEMA = "mergenet.cifar_resize_checkpoint_parity.v1"
CHECKPOINT_PARITY_RUNNER = SCRIPT_DIR / "post_training_parity.py"
CHECKPOINT_PARITY_MODELS = ("mn_l2", "mn_l4")
CHECKPOINT_PARITY_SAMPLES = 10_000
CHECKPOINT_PARITY_CLASSES = 100
CHECKPOINT_PARITY_MAX_CORRECT_DELTA = 5
CHECKPOINT_PARITY_MAX_DELTA_PP = 0.05
CHECKPOINT_PARITY_GENERIC = "alternating_per_layer"
CHECKPOINT_PARITY_FAST = "alternating_per_layer_fast"
EXPECTED_RELEASE_ENVIRONMENT = {
    "python": "3.10",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "timm": "0.9.11",
    "flash_attn": "2.7.4.post1",
}
EXPECTED_RELEASE_RUNTIME_ENV = {
    "OPENTOME_MERGENET_IMPL": "new",
    "TIMM_FUSED_ATTN": "1",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integer; got {value!r}") from exc


def model_id(entry: Any) -> str:
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, Mapping):
        value = entry.get("id") or entry.get("model_id") or entry.get("name")
    else:
        value = None
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid model entry: {entry!r}")
    return value


def resize_value(entry: Any) -> int:
    if isinstance(entry, Mapping):
        entry = entry.get("size", entry.get("resize"))
    value = _as_int(entry, "resize")
    if value <= 0:
        raise ValueError(f"resize must be positive; got {value}")
    return value


def protocol_matrix(protocol: Mapping[str, Any]) -> tuple[list[str], list[int], list[int]]:
    models = [model_id(entry) for entry in protocol.get("models", [])]
    resizes = [resize_value(entry) for entry in protocol.get("resizes", [])]
    seeds = [_as_int(seed, "seed") for seed in protocol.get("seeds", [])]
    if not models or not resizes or not seeds:
        raise ValueError("protocol models/resizes/seeds cannot be empty")
    if len(set(models)) != len(models) or len(set(resizes)) != len(resizes) or len(set(seeds)) != len(seeds):
        raise ValueError("protocol models/resizes/seeds must each be unique")
    return models, resizes, seeds


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sample_stats(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    count = len(data)
    if not data:
        return {"n": 0, "mean": None, "sample_sd": None, "values": []}
    mean = statistics.fmean(data)
    sd = statistics.stdev(data) if count >= 2 else None
    return {"n": count, "mean": mean, "sample_sd": sd, "values": data}


def formatted_stats(stats: Mapping[str, Any], digits: int = 2) -> str:
    mean = stats.get("mean")
    sd = stats.get("sample_sd")
    if mean is None:
        return "—"
    if sd is None:
        return f"{float(mean):.{digits}f} (n={stats.get('n', 0)})"
    return f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"


def job_key(model: str, resize: int, seed: int) -> str:
    return f"{model}__r{resize}__s{seed}"


def canonical_job_dir(root: Path, model: str, resize: int, seed: int) -> Path:
    return root / "runs" / model / f"r{resize}" / f"seed{seed}"


def find_summary(directory: Path) -> Path | None:
    preferred = (directory / "summary.csv", directory / "run" / "summary.csv")
    for path in preferred:
        if path.is_file():
            return path
    if not directory.is_dir():
        return None
    candidates = sorted(
        directory.rglob("summary.csv"),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_accuracy_run(
    root: Path,
    state_jobs: Mapping[str, Any],
    model: str,
    resize: int,
    seed: int,
    target_epoch: int,
) -> dict[str, Any]:
    directory = canonical_job_dir(root, model, resize, seed)
    summary = find_summary(directory)
    base = {
        "model_id": model,
        "resize": resize,
        "seed": seed,
        "job_dir": str(directory),
        "summary": str(summary) if summary else None,
        "target_epoch": target_epoch,
        "last_epoch": None,
        "ema_top1": None,
        "metric_key": None,
        "artifact_verified": False,
        "status": "missing",
    }
    if summary is None:
        return base
    try:
        with summary.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return {**base, "status": "invalid", "error": str(exc)}

    parsed: list[tuple[int, dict[str, str], str, float]] = []
    for row in rows:
        try:
            epoch = int(float(str(row.get("epoch", ""))))
        except (TypeError, ValueError):
            continue
        metric_key = next(
            (
                key
                for key in (
                    "eval_top1_ema",
                    "eval_ema_top1",
                    "ema_top1",
                    "eval_top1",
                    "top1",
                )
                if finite_float(row.get(key)) is not None
            ),
            None,
        )
        if metric_key is None:
            continue
        parsed.append((epoch, row, metric_key, float(row[metric_key])))
    if not parsed:
        return {**base, "status": "invalid", "error": "no finite top1 rows"}

    last_epoch = max(row[0] for row in parsed)
    best = max(parsed, key=lambda row: row[3])
    target_rows = [row for row in parsed if row[0] == target_epoch]
    result = {
        **base,
        "last_epoch": last_epoch,
        "best_epoch": best[0],
        "best_top1": best[3],
        "status": "partial",
    }
    if not target_rows:
        return result
    target = target_rows[-1]
    result.update(ema_top1=target[3], metric_key=target[2], status="epoch_complete_unverified")

    state_entry = state_jobs.get(job_key(model, resize, seed), {})
    state_verified = isinstance(state_entry, Mapping) and state_entry.get("status") == "completed"
    marker_verified = False
    marker_candidates = [directory / "completion.json"]
    if directory.is_dir():
        marker_candidates.extend(directory.glob("*/completion.json"))
    for marker in marker_candidates:
        if not marker.is_file():
            continue
        try:
            value = load_json(marker)
            marker_verified = (
                value.get("status") == "complete"
                and int(value.get("epoch")) == target_epoch
                and value.get("ema") is True
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            marker_verified = False
        if marker_verified:
            break
    result["artifact_verified"] = state_verified or marker_verified
    result["verified_by"] = "campaign_state" if state_verified else "completion_marker" if marker_verified else None
    if result["artifact_verified"]:
        result["status"] = "complete"
    return result


def read_campaign_state(root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = root / "state" / "campaign_state.json"
    if not path.is_file():
        return {"available": False, "path": str(path)}, {}
    try:
        state = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "path": str(path), "error": str(exc)}, {}
    if not isinstance(state, Mapping):
        return {"available": False, "path": str(path), "error": "not an object"}, {}
    jobs = state.get("jobs", {})
    return (
        {
            "available": True,
            "path": str(path),
            "phase": state.get("phase"),
            "updated_at": state.get("updated_at"),
        },
        jobs if isinstance(jobs, Mapping) else {},
    )


def aggregate_accuracy(
    protocol: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    models, resizes, seeds = protocol_matrix(protocol)
    training = protocol.get("training", {})
    epochs = _as_int(training.get("epochs", 200) if isinstance(training, Mapping) else 200, "epochs")
    target_epoch = epochs - 1
    state_info, state_jobs = read_campaign_state(root)
    runs = [
        read_accuracy_run(root, state_jobs, model, resize, seed, target_epoch)
        for resize in resizes
        for model in models
        for seed in seeds
    ]
    by_key = {(row["model_id"], row["resize"], row["seed"]): row for row in runs}

    summary: list[dict[str, Any]] = []
    for resize in resizes:
        for model in models:
            group = [by_key[(model, resize, seed)] for seed in seeds]
            complete = [row for row in group if row["status"] == "complete"]
            stats = sample_stats(row["ema_top1"] for row in complete)
            summary.append(
                {
                    "model_id": model,
                    "resize": resize,
                    "metric": "epoch_199_ema_top1",
                    "expected_seeds": len(seeds),
                    "complete_seeds": [row["seed"] for row in complete],
                    "missing_or_unverified_seeds": [row["seed"] for row in group if row["status"] != "complete"],
                    "complete": len(complete) == len(seeds),
                    **stats,
                }
            )

    rules = protocol.get("decision_rules", {})
    lambda4 = rules.get("lambda4", {}) if isinstance(rules, Mapping) else {}
    baseline = (
        lambda4.get("baseline_model")
        if isinstance(lambda4, Mapping)
        else None
    )
    if not baseline:
        baseline = next((model for model in models if "deit" in model.lower()), models[0])
    comparisons = [model for model in models if model != baseline]
    paired: list[dict[str, Any]] = []
    for resize in resizes:
        for candidate in comparisons:
            pairs = []
            for seed in seeds:
                candidate_row = by_key[(candidate, resize, seed)]
                baseline_row = by_key.get((baseline, resize, seed))
                if (
                    baseline_row
                    and candidate_row["status"] == "complete"
                    and baseline_row["status"] == "complete"
                ):
                    pairs.append(
                        {
                            "seed": seed,
                            "candidate_top1": candidate_row["ema_top1"],
                            "baseline_top1": baseline_row["ema_top1"],
                            "delta_pp": candidate_row["ema_top1"] - baseline_row["ema_top1"],
                        }
                    )
            stats = sample_stats(pair["delta_pp"] for pair in pairs)
            paired.append(
                {
                    "candidate_model": candidate,
                    "baseline_model": baseline,
                    "resize": resize,
                    "metric": "paired_accuracy_delta_pp",
                    "expected_pairs": len(seeds),
                    "complete": len(pairs) == len(seeds),
                    "pairs": pairs,
                    **stats,
                }
            )
    return runs, summary, paired, state_info


EFFICIENCY_FIELDS = (
    "throughput_img_s",
    "step_time_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "params",
)


def expected_efficiency_items() -> dict[str, tuple[str, int, str]]:
    modes = {
        "deit_s8": ("train", "infer"),
        "mn_l2": ("train_random_per_sample", "infer_generic", "infer_fast", "logits_parity"),
        "mn_l4": ("train_random_per_sample", "infer_generic", "infer_fast", "logits_parity"),
    }
    return {
        f"{model}_r{resize}_{mode}": (model, resize, mode)
        for resize in (160, 192, 224, 256, 320)
        for model, model_modes in modes.items()
        for mode in model_modes
    }


def efficiency_document_is_formal(document: Mapping[str, Any]) -> bool:
    if (
        document.get("complete") is not True
        or document.get("all_success") is not True
        or document.get("canonical_environment") is not True
        or document.get("noncanonical") is True
    ):
        return False
    try:
        gpu = int(document.get("physical_gpu"))
    except (TypeError, ValueError):
        return False
    gpu_uuid = document.get("gpu_uuid")
    items = document.get("items")
    expected = expected_efficiency_items()
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        return False
    if not isinstance(items, list) or len(items) != len(expected):
        return False
    indexed = {
        item.get("item_id"): item for item in items if isinstance(item, Mapping)
    }
    if len(indexed) != len(items) or set(indexed) != set(expected):
        return False
    for item_id, (model, resize, mode) in expected.items():
        item = indexed[item_id]
        try:
            identity_ok = (
                item.get("model_id") == model
                and int(item.get("resize")) == resize
                and item.get("mode") == mode
            )
        except (TypeError, ValueError):
            return False
        environment = item.get("environment")
        if (
            not identity_ok
            or item.get("success") is not True
            or not isinstance(environment, Mapping)
            or environment.get("canonical") is not True
            or environment.get("noncanonical") is True
            or environment.get("gpu_uuid") != gpu_uuid
        ):
            return False
        try:
            if int(environment.get("physical_gpu")) != gpu:
                return False
        except (TypeError, ValueError):
            return False
        if mode == "logits_parity":
            if item.get("allclose") is not True:
                return False
        elif item.get("timing_valid") is not True:
            return False
    return True


def load_efficiency(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for path in sorted((root / "efficiency").glob("gpu*.json")):
        try:
            document = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            documents.append({"path": str(path), "complete": False, "error": str(exc)})
            continue
        if not isinstance(document, Mapping):
            documents.append({"path": str(path), "complete": False, "error": "not an object"})
            continue
        gpu = finite_float(document.get("physical_gpu"))
        gpu_id = int(gpu) if gpu is not None else None
        complete = efficiency_document_is_formal(document)
        documents.append(
            {
                "path": str(path),
                "physical_gpu": gpu_id,
                "complete": complete,
                "item_count": len(document.get("items", [])) if isinstance(document.get("items"), list) else 0,
            }
        )
        raw_items = document.get("items", [])
        if not isinstance(raw_items, list):
            continue
        # Incremental files may retain an older item with the same item_id.  The
        # last occurrence is authoritative.
        dedup: dict[str, Mapping[str, Any]] = {}
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("item_id", f"row{index}"))
            dedup[key] = item
        for item in dedup.values():
            try:
                resize = int(item.get("resize"))
            except (TypeError, ValueError):
                continue
            raw_mode = str(item.get("mode", "")).lower()
            model = item.get("model_id")
            if not isinstance(model, str) or not model:
                continue
            if raw_mode == "logits_parity":
                items.append(
                    {
                        "physical_gpu": gpu_id,
                        "source_complete": complete,
                        "source": str(path),
                        "model_id": model,
                        "resize": resize,
                        "mode": "logits_parity",
                        "success": item.get("success") is True,
                        "allclose": item.get("allclose") is True,
                        "max_abs_diff": finite_float(item.get("max_abs_diff")),
                        "mean_abs_diff": finite_float(item.get("mean_abs_diff")),
                        "rms_diff": finite_float(item.get("rms_diff")),
                        "max_relative_diff": finite_float(item.get("max_relative_diff")),
                        "cosine_similarity": finite_float(item.get("cosine_similarity")),
                        "argmax_agreement": finite_float(item.get("argmax_agreement")),
                    }
                )
                continue
            if not item.get("success"):
                continue
            if raw_mode in {"train", "training", "train_random_per_sample"}:
                mode = "train"
            elif raw_mode in {"infer", "inference", "eval", "evaluation", "infer_generic"}:
                mode = "infer_generic"
            elif raw_mode == "infer_fast":
                mode = "infer_fast"
            elif raw_mode.startswith("train") and "fast" in raw_mode:
                mode = "train_fast"
            else:
                continue
            # Successful output collected while the host gate was violated is
            # audit data, not formal timing evidence.
            timing_valid = item.get("timing_valid") is True
            row: dict[str, Any] = {
                "physical_gpu": gpu_id,
                "source_complete": complete,
                "timing_valid": timing_valid,
                "source": str(path),
                "model_id": model,
                "resize": resize,
                "mode": mode,
                "batch_size": item.get("batch_size"),
            }
            for field in EFFICIENCY_FIELDS:
                row[field] = finite_float(item.get(field))
            items.append(row)
    return documents, items


def aggregate_efficiency(
    protocol: Mapping[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    models, resizes, _ = protocol_matrix(protocol)
    documents, items = load_efficiency(root)
    timed_items = [row for row in items if row["mode"] != "logits_parity"]
    parity_items = [row for row in items if row["mode"] == "logits_parity"]
    raw_summary: list[dict[str, Any]] = []
    observed_modes = [
        mode
        for mode in ("train", "train_fast", "infer_generic", "infer_fast")
        if any(row["mode"] == mode for row in timed_items)
    ]
    for resize in resizes:
        for mode in observed_modes:
            for model in models:
                group = [
                    row
                    for row in timed_items
                    if row["model_id"] == model and row["resize"] == resize and row["mode"] == mode
                ]
                entry: dict[str, Any] = {
                    "model_id": model,
                    "resize": resize,
                    "mode": mode,
                    "physical_gpus": sorted({row["physical_gpu"] for row in group if row["physical_gpu"] is not None}),
                    "complete_physical_gpus": sorted(
                        {
                            row["physical_gpu"]
                            for row in group
                            if row["physical_gpu"] is not None
                            and row["source_complete"]
                            and row["timing_valid"]
                        }
                    ),
                    "provisional": any(
                        not row["source_complete"] or not row["timing_valid"]
                        for row in group
                    ),
                }
                entry["provisional_metrics"] = {}
                for field in EFFICIENCY_FIELDS:
                    entry[field] = sample_stats(
                        row[field]
                        for row in group
                        if row[field] is not None
                        and row["timing_valid"]
                        and row["source_complete"]
                    )
                    entry["provisional_metrics"][field] = sample_stats(
                        row[field]
                        for row in group
                        if row[field] is not None and row["timing_valid"]
                    )
                raw_summary.append(entry)

    rules = protocol.get("decision_rules", {})
    lambda4 = rules.get("lambda4", {}) if isinstance(rules, Mapping) else {}
    baseline = lambda4.get("baseline_model") if isinstance(lambda4, Mapping) else None
    if not baseline:
        baseline = next((model for model in models if "deit" in model.lower()), models[0])
    comparisons = [model for model in models if model != baseline]
    item_index = {
        (row["physical_gpu"], row["model_id"], row["resize"], row["mode"]): row
        for row in timed_items
        if row["physical_gpu"] is not None
        and row["source_complete"]
        and row["timing_valid"]
    }
    paired_ratios: list[dict[str, Any]] = []
    ratio_fields = {
        "throughput_ratio": "throughput_img_s",
        "step_time_ratio": "step_time_ms",
        "peak_allocated_ratio": "peak_allocated_mib",
        "peak_reserved_ratio": "peak_reserved_mib",
        "params_ratio": "params",
    }
    all_gpus = sorted({row["physical_gpu"] for row in timed_items if row["physical_gpu"] is not None})
    for resize in resizes:
        for mode in observed_modes:
            for candidate in comparisons:
                pairs = []
                for gpu in all_gpus:
                    candidate_row = item_index.get((gpu, candidate, resize, mode))
                    # DeiT has one inference/train implementation; reuse that
                    # same-card baseline for a MergeNet fast-path comparison.
                    baseline_mode = (
                        "infer_generic"
                        if mode == "infer_fast"
                        else "train"
                        if mode == "train_fast"
                        else mode
                    )
                    baseline_row = item_index.get((gpu, baseline, resize, baseline_mode))
                    if not candidate_row or not baseline_row:
                        continue
                    pair: dict[str, Any] = {"physical_gpu": gpu}
                    for ratio_name, source_field in ratio_fields.items():
                        numerator = candidate_row.get(source_field)
                        denominator = baseline_row.get(source_field)
                        pair[ratio_name] = (
                            numerator / denominator
                            if numerator is not None and denominator not in (None, 0)
                            else None
                        )
                    pairs.append(pair)
                entry = {
                    "candidate_model": candidate,
                    "baseline_model": baseline,
                    "resize": resize,
                    "mode": mode,
                    "physical_gpus": [pair["physical_gpu"] for pair in pairs],
                }
                for ratio_name in ratio_fields:
                    entry[ratio_name] = sample_stats(
                        pair[ratio_name] for pair in pairs if pair[ratio_name] is not None
                    )
                entry["per_gpu"] = pairs
                paired_ratios.append(entry)
    required_gpu_count = int(
        protocol.get("efficiency", {}).get("required_gpu_count", 8)
        if isinstance(protocol.get("efficiency"), Mapping)
        else 8
    )
    parity_summary: list[dict[str, Any]] = []
    parity_fields = (
        "max_abs_diff",
        "mean_abs_diff",
        "rms_diff",
        "max_relative_diff",
        "cosine_similarity",
        "argmax_agreement",
    )
    for resize in resizes:
        for model in (candidate for candidate in models if candidate != baseline):
            observed = [
                row
                for row in parity_items
                if row["model_id"] == model and row["resize"] == resize
            ]
            formal = [row for row in observed if row["source_complete"]]
            entry: dict[str, Any] = {
                "model_id": model,
                "resize": resize,
                "expected_gpu_count": required_gpu_count,
                "observed_gpus": sorted(
                    row["physical_gpu"] for row in observed if row["physical_gpu"] is not None
                ),
                "formal_gpus": sorted(
                    row["physical_gpu"] for row in formal if row["physical_gpu"] is not None
                ),
                "complete": (
                    len(formal) >= required_gpu_count
                    and all(row["success"] and row["allclose"] for row in formal)
                ),
                "allclose_all": (
                    all(row["allclose"] for row in formal) if formal else None
                ),
                "provisional_allclose_all": (
                    all(row["allclose"] for row in observed) if observed else None
                ),
                "metrics": {},
                "provisional_metrics": {},
            }
            for field in parity_fields:
                entry["metrics"][field] = sample_stats(
                    row[field] for row in formal if row[field] is not None
                )
                entry["provisional_metrics"][field] = sample_stats(
                    row[field] for row in observed if row[field] is not None
                )
            max_values = [row["max_abs_diff"] for row in formal if row["max_abs_diff"] is not None]
            provisional_max_values = [
                row["max_abs_diff"] for row in observed if row["max_abs_diff"] is not None
            ]
            entry["worst_max_abs_diff"] = max(max_values) if max_values else None
            entry["provisional_worst_max_abs_diff"] = (
                max(provisional_max_values) if provisional_max_values else None
            )
            parity_summary.append(entry)
    return documents, raw_summary, paired_ratios, parity_summary


OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def evaluate_lambda4_gate(
    protocol: Mapping[str, Any],
    accuracy_paired: Sequence[Mapping[str, Any]],
    efficiency_paired: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rules = protocol.get("decision_rules", {})
    rule = rules.get("lambda4", {}) if isinstance(rules, Mapping) else {}
    if not isinstance(rule, Mapping) or not rule:
        return {"status": "NOT_CONFIGURED", "reason": "decision_rules.lambda4 missing"}
    candidate = str(rule.get("candidate_model", "mn_l4"))
    baseline = str(rule.get("baseline_model", "deit_s8"))
    required_seed_value = rule.get("required_complete_seeds", 3)
    required_seeds = (
        len(required_seed_value)
        if isinstance(required_seed_value, list)
        else _as_int(required_seed_value, "required seeds")
    )
    required_gpus = _as_int(rule.get("required_gpu_count", 8), "required GPUs")
    conditions = rule.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        return {"status": "NOT_CONFIGURED", "reason": "lambda4 conditions missing"}

    accuracy_index = {
        (row.get("candidate_model"), row.get("baseline_model"), row.get("resize")): row
        for row in accuracy_paired
    }
    efficiency_index = {
        (
            row.get("candidate_model"),
            row.get("baseline_model"),
            row.get("resize"),
            row.get("mode"),
        ): row
        for row in efficiency_paired
    }
    details: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            raise ValueError(f"lambda4 condition {condition_index} must be an object")
        metric = str(condition.get("metric"))
        op_name = str(condition.get("op"))
        if op_name not in OPS:
            raise ValueError(f"unsupported decision operator {op_name!r}")
        threshold = float(condition.get("threshold"))
        requested_resizes = condition.get("resizes", [])
        if requested_resizes == "all":
            _, requested_resizes, _ = protocol_matrix(protocol)
        if not isinstance(requested_resizes, list) or not requested_resizes:
            raise ValueError(f"condition {metric} must preregister non-empty resizes")
        per_resize = []
        for resize_raw in requested_resizes:
            resize = int(resize_raw)
            stats: Mapping[str, Any] | None = None
            evidence_n = 0
            required_n = required_gpus
            if metric == "paired_accuracy_delta_pp":
                row = accuracy_index.get((candidate, baseline, resize))
                stats = row if row else None
                evidence_n = int(row.get("n", 0)) if row else 0
                required_n = required_seeds
            else:
                if metric.startswith("train_fast_"):
                    mode, ratio_name = "train_fast", metric.removeprefix("train_fast_")
                elif metric.startswith("train_"):
                    mode, ratio_name = "train", metric.removeprefix("train_")
                elif metric.startswith("infer_fast_"):
                    mode, ratio_name = "infer_fast", metric.removeprefix("infer_fast_")
                elif metric.startswith("infer_generic_"):
                    mode, ratio_name = "infer_generic", metric.removeprefix("infer_generic_")
                elif metric.startswith("infer_"):
                    mode, ratio_name = "infer_generic", metric.removeprefix("infer_")
                else:
                    mode, ratio_name = None, metric
                row = efficiency_index.get((candidate, baseline, resize, mode)) if mode else None
                ratio_stats = row.get(ratio_name) if row else None
                stats = ratio_stats if isinstance(ratio_stats, Mapping) else None
                evidence_n = int(stats.get("n", 0)) if stats else 0
            mean = stats.get("mean") if stats else None
            enough = evidence_n >= required_n
            observed_pass = OPS[op_name](float(mean), threshold) if enough and mean is not None else None
            status = "PASS" if observed_pass is True else "FAIL" if observed_pass is False else "INCOMPLETE"
            per_resize.append(
                {
                    "resize": resize,
                    "value": mean,
                    "sample_sd": stats.get("sample_sd") if stats else None,
                    "n": evidence_n,
                    "required_n": required_n,
                    "op": op_name,
                    "threshold": threshold,
                    "status": status,
                }
            )
        condition_status = (
            "INCOMPLETE"
            if any(row["status"] == "INCOMPLETE" for row in per_resize)
            else "FAIL"
            if any(row["status"] == "FAIL" for row in per_resize)
            else "PASS"
        )
        details.append(
            {
                "metric": metric,
                "status": condition_status,
                "per_resize": per_resize,
            }
        )
    overall = (
        "INCOMPLETE"
        if any(row["status"] == "INCOMPLETE" for row in details)
        else "FAIL"
        if any(row["status"] == "FAIL" for row in details)
        else "PASS"
    )
    return {
        "status": overall,
        "candidate_model": candidate,
        "baseline_model": baseline,
        "required_complete_seeds": required_seeds,
        "required_gpu_count": required_gpus,
        "conditions": details,
        "note": "Inference is reported separately and affects the gate only if explicitly preregistered.",
    }


def checkpoint_parity_tasks(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    models, resizes, seeds = protocol_matrix(protocol)
    if tuple(model for model in models if model in CHECKPOINT_PARITY_MODELS) != CHECKPOINT_PARITY_MODELS:
        raise ValueError("protocol lacks the locked mn_l2/mn_l4 checkpoint parity models")
    resize_batches = {
        resize_value(entry): _as_int(entry.get("micro_batch"), "micro_batch")
        for entry in protocol.get("resizes", [])
        if isinstance(entry, Mapping)
    }
    if tuple(resizes) != (160, 192, 224, 256, 320) or tuple(seeds) != (42, 43, 44):
        raise ValueError("protocol parity resize/seed matrix drift")
    return [
        {
            "task_id": job_key(model, resize, seed),
            "model_id": model,
            "resize": resize,
            "seed": seed,
            "validation_batch_size": resize_batches[resize],
            "filename": f"{model}_r{resize}_seed{seed}.json",
        }
        for resize in resizes
        for model in CHECKPOINT_PARITY_MODELS
        for seed in seeds
    ]


def _snapshot_runtime_fingerprint(root: Path, protocol_path: Path) -> dict[str, Any]:
    snapshot_path = root / "runtime" / "snapshot_manifest.json"
    runtime_root = root / "runtime" / "imagenet_longtrain_v1"
    immutable_protocol = root / "runtime" / "cifar_resize_20260810" / "protocol.json"
    if protocol_path != immutable_protocol.resolve():
        raise ValueError("final parity aggregation requires the immutable campaign protocol")
    if not snapshot_path.is_file() or snapshot_path.is_symlink():
        raise ValueError("immutable snapshot manifest is missing or is a symlink")
    snapshot = load_json(snapshot_path)
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("runtime_files"), Mapping):
        raise ValueError("invalid immutable snapshot manifest")
    expected_files = dict(snapshot["runtime_files"])
    actual_files: dict[str, str] = {}
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise ValueError("immutable delivery runtime missing or symlinked")
    for path in sorted(runtime_root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"immutable runtime contains symlink: {path}")
        if path.is_file():
            actual_files[str(path.relative_to(runtime_root))] = sha256_file(path)
        elif not path.is_dir():
            raise ValueError(f"unsupported immutable runtime entry: {path}")
    if actual_files != expected_files:
        raise ValueError("immutable delivery runtime no longer matches snapshot hashes")
    protocol_file_sha = sha256_file(protocol_path)
    harness_files = snapshot.get("harness_files")
    if not isinstance(harness_files, Mapping) or harness_files.get("protocol.json") != protocol_file_sha:
        raise ValueError("immutable protocol no longer matches snapshot hashes")
    return {
        "runtime_tree_sha256": hashlib.sha256(canonical_json(actual_files)).hexdigest(),
        "snapshot_manifest_sha256": sha256_file(snapshot_path),
        "snapshot_bundle_sha256": snapshot.get("bundle_sha256"),
        "protocol_file_sha256": protocol_file_sha,
    }


def _close_float(actual: Any, expected: float, name: str, tolerance: float = 1e-9) -> float:
    value = finite_float(actual)
    if value is None or not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name}={actual!r}, expected {expected!r}")
    return value


def _validate_checkpoint_parity_report(
    path: Path,
    expected_task: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    root: Path,
    immutable: Mapping[str, Any],
    runner_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("report is missing, non-regular, or symlinked")
    report = load_json(path)
    if not isinstance(report, Mapping):
        raise ValueError("report root is not an object")
    if report.get("schema_version") != CHECKPOINT_PARITY_SCHEMA:
        raise ValueError("report schema mismatch")
    if report.get("runner_revision") != 1 or report.get("status") != "complete":
        raise ValueError("report is not a revision-1 complete result")
    task = report.get("task")
    expected_identity = {
        key: expected_task[key]
        for key in ("task_id", "model_id", "resize", "seed", "validation_batch_size")
    }
    if not isinstance(task, Mapping) or dict(task) != expected_identity:
        raise ValueError("report task identity mismatch")
    protocol_file_sha = sha256_file(protocol_path)
    protocol_canonical_sha = hashlib.sha256(canonical_json(protocol)).hexdigest()
    identity = report.get("identity")
    expected_report_identity = {
        "runner_revision": 1,
        "runner_sha256": runner_sha256,
        "task": expected_identity,
        "protocol_file_sha256": protocol_file_sha,
        "protocol_canonical_sha256": protocol_canonical_sha,
        "runtime_tree_sha256": immutable["runtime_tree_sha256"],
    }
    if not isinstance(identity, Mapping):
        raise ValueError("report identity missing")
    for key, expected in expected_report_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"report identity {key} mismatch")

    environment = report.get("environment")
    if not isinstance(environment, Mapping) or environment.get("canonical") is not True:
        raise ValueError("report environment is not canonical")
    if protocol.get("expected_environment") != EXPECTED_RELEASE_ENVIRONMENT:
        raise ValueError("protocol expected_environment differs from release lock")
    if protocol.get("expected_runtime_env") != EXPECTED_RELEASE_RUNTIME_ENV:
        raise ValueError("protocol expected_runtime_env differs from release lock")
    if environment.get("versions") != EXPECTED_RELEASE_ENVIRONMENT:
        raise ValueError("report dependency versions differ from release lock")
    if environment.get("runtime_env") != EXPECTED_RELEASE_RUNTIME_ENV:
        raise ValueError("report runtime environment differs from release lock")
    gpu = environment.get("gpu")
    if not isinstance(gpu, Mapping) or not str(gpu.get("uuid", "")).startswith("GPU-"):
        raise ValueError("report lacks a physical GPU UUID")
    try:
        int(gpu.get("physical_index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("report lacks a physical GPU index") from exc

    runtime = report.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("tree_sha256") != immutable["runtime_tree_sha256"]
        or runtime.get("snapshot_manifest_sha256") != immutable["snapshot_manifest_sha256"]
        or runtime.get("snapshot_bundle_sha256") != immutable["snapshot_bundle_sha256"]
    ):
        raise ValueError("report immutable runtime identity mismatch")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("report checkpoint evidence missing")
    expected_checkpoint = canonical_job_dir(
        root,
        str(expected_task["model_id"]),
        int(expected_task["resize"]),
        int(expected_task["seed"]),
    ) / "last.pth.tar"
    if expected_checkpoint.is_symlink() or not expected_checkpoint.is_file():
        raise ValueError("current canonical checkpoint missing or symlinked")
    checkpoint_sha = sha256_file(expected_checkpoint)
    if (
        checkpoint.get("checkpoint_path") != str(expected_checkpoint.resolve())
        or checkpoint.get("checkpoint_sha256") != checkpoint_sha
        or identity.get("checkpoint_sha256") != checkpoint_sha
        or checkpoint.get("checkpoint_epoch") != 199
        or checkpoint.get("ema_state_key") != "state_dict_ema"
        or checkpoint.get("strict_state_dict_load") is not True
    ):
        raise ValueError("checkpoint path/SHA/epoch/EMA identity mismatch")
    completion_path = expected_checkpoint.parent / "completion.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError("current completion marker missing or symlinked")
    completion = load_json(completion_path)
    if not isinstance(completion, Mapping):
        raise ValueError("completion marker is not an object")
    if (
        completion.get("status") != "complete"
        or completion.get("epoch") != 199
        or completion.get("ema") is not True
        or completion.get("model_id") != expected_task["model_id"]
        or completion.get("resize") != expected_task["resize"]
        or completion.get("seed") != expected_task["seed"]
        or completion.get("checkpoint_sha256") != checkpoint_sha
        or checkpoint.get("completion_sha256") != sha256_file(completion_path)
    ):
        raise ValueError("current completion marker identity mismatch")

    data = report.get("data")
    if (
        not isinstance(data, Mapping)
        or data.get("test_md5") != "f0ef6b0ae62326f3e7ffdfab6717acfc"
        or data.get("expected_samples") != CHECKPOINT_PARITY_SAMPLES
    ):
        raise ValueError("report CIFAR-100 data identity mismatch")
    test_path = Path(str(data.get("test_path", "")))
    if test_path.is_symlink() or not test_path.is_file() or sha256_file(test_path) != data.get("test_sha256"):
        raise ValueError("current CIFAR-100 test file differs from report")

    evaluation = report.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("report evaluation missing")
    if (
        evaluation.get("dataset") != "CIFAR100"
        or evaluation.get("split") != "test"
        or evaluation.get("sample_count") != CHECKPOINT_PARITY_SAMPLES
        or evaluation.get("class_count") != CHECKPOINT_PARITY_CLASSES
        or evaluation.get("loader_deterministic") is not True
        or evaluation.get("loader_shared_between_modes") is not True
        or evaluation.get("generic_grouping") != CHECKPOINT_PARITY_GENERIC
        or evaluation.get("fast_grouping") != CHECKPOINT_PARITY_FAST
        or evaluation.get("grouping_seed") != 0
        or evaluation.get("amp") is not True
        or evaluation.get("amp_dtype") != "float16"
        or evaluation.get("validation_batch_size") != expected_task["validation_batch_size"]
    ):
        raise ValueError("evaluation protocol mismatch")
    try:
        generic_correct = int(evaluation.get("generic_correct"))
        fast_correct = int(evaluation.get("fast_correct"))
        agreement_count = int(evaluation.get("argmax_agreement_count"))
        mismatch_count = int(evaluation.get("argmax_mismatch_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation integer counts are malformed") from exc
    for name, count in (
        ("generic_correct", generic_correct),
        ("fast_correct", fast_correct),
        ("agreement_count", agreement_count),
        ("mismatch_count", mismatch_count),
    ):
        if not 0 <= count <= CHECKPOINT_PARITY_SAMPLES:
            raise ValueError(f"{name} out of range")
    if agreement_count + mismatch_count != CHECKPOINT_PARITY_SAMPLES:
        raise ValueError("argmax agreement/mismatch counts do not sum to 10000")
    generic_top1 = generic_correct * 100.0 / CHECKPOINT_PARITY_SAMPLES
    fast_top1 = fast_correct * 100.0 / CHECKPOINT_PARITY_SAMPLES
    delta_pp = (fast_correct - generic_correct) * 100.0 / CHECKPOINT_PARITY_SAMPLES
    _close_float(evaluation.get("generic_top1"), generic_top1, "generic_top1")
    _close_float(evaluation.get("fast_top1"), fast_top1, "fast_top1")
    _close_float(evaluation.get("top1_delta_pp"), delta_pp, "top1_delta_pp")
    _close_float(evaluation.get("abs_top1_delta_pp"), abs(delta_pp), "abs_top1_delta_pp")
    argmax_agreement = agreement_count / CHECKPOINT_PARITY_SAMPLES
    _close_float(evaluation.get("argmax_agreement"), argmax_agreement, "argmax_agreement")
    max_diff = finite_float(evaluation.get("max_abs_logit_diff"))
    mean_diff = finite_float(evaluation.get("mean_abs_logit_diff"))
    if max_diff is None or mean_diff is None or min(max_diff, mean_diff) < 0 or mean_diff > max_diff + 1e-12:
        raise ValueError("invalid max/mean absolute logit differences")
    gate_pass = (
        abs(fast_correct - generic_correct) <= CHECKPOINT_PARITY_MAX_CORRECT_DELTA
        and abs(delta_pp) <= CHECKPOINT_PARITY_MAX_DELTA_PP + 1e-12
    )
    gate = report.get("gate")
    expected_gate_status = "PASS" if gate_pass else "FAIL"
    if (
        not isinstance(gate, Mapping)
        or gate.get("pass") is not gate_pass
        or gate.get("status") != expected_gate_status
        or finite_float(gate.get("threshold_pp")) != CHECKPOINT_PARITY_MAX_DELTA_PP
        or gate.get("max_correct_count_difference") != CHECKPOINT_PARITY_MAX_CORRECT_DELTA
        or gate.get("correct_count_difference") != fast_correct - generic_correct
    ):
        raise ValueError("reported gate disagrees with recomputed top-1 condition")
    return {
        **expected_identity,
        "path": str(path),
        "status": "complete",
        "gate_status": expected_gate_status,
        "gate_pass": gate_pass,
        "gpu_index": int(gpu["physical_index"]),
        "gpu_uuid": str(gpu["uuid"]),
        "checkpoint_path": str(expected_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "generic_correct": generic_correct,
        "fast_correct": fast_correct,
        "generic_top1": generic_top1,
        "fast_top1": fast_top1,
        "top1_delta_pp": delta_pp,
        "abs_top1_delta_pp": abs(delta_pp),
        "argmax_agreement_count": agreement_count,
        "argmax_mismatch_count": mismatch_count,
        "argmax_agreement": argmax_agreement,
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "fast_vs_training_summary_delta_pp": finite_float(
            evaluation.get("fast_vs_training_summary_delta_pp")
        ),
    }


def load_checkpoint_parity(
    protocol: Mapping[str, Any], protocol_path: Path, root: Path
) -> dict[str, Any]:
    gate = protocol.get("post_training_release_gate")
    if not isinstance(gate, Mapping) or gate.get("required_for_final_release") is not True:
        return {
            "gate_status": "INCOMPLETE",
            "error": "post_training_release_gate is missing or not mandatory",
            "expected_runs": 30,
            "valid_complete_runs": 0,
            "passed_runs": 0,
            "failed_runs": 0,
            "missing_runs": [],
            "invalid_runs": [],
            "runs": [],
        }
    if not CHECKPOINT_PARITY_RUNNER.is_file():
        return {
            "gate_status": "INCOMPLETE",
            "error": f"parity runner missing: {CHECKPOINT_PARITY_RUNNER}",
            "expected_runs": 30,
            "valid_complete_runs": 0,
            "passed_runs": 0,
            "failed_runs": 0,
            "missing_runs": [],
            "invalid_runs": [],
            "runs": [],
        }
    tasks = checkpoint_parity_tasks(protocol)
    out_dir = root / "post_training_parity"
    try:
        immutable = _snapshot_runtime_fingerprint(root, protocol_path)
    except Exception as exc:
        return {
            "gate_status": "INCOMPLETE",
            "error": f"immutable campaign validation failed: {type(exc).__name__}: {exc}",
            "report_dir": str(out_dir),
            "expected_runs": len(tasks),
            "valid_complete_runs": 0,
            "passed_runs": 0,
            "failed_runs": 0,
            "missing_runs": [task["task_id"] for task in tasks],
            "invalid_runs": [],
            "runs": [],
        }
    runner_sha = sha256_file(CHECKPOINT_PARITY_RUNNER)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for task in tasks:
        path = out_dir / task["filename"]
        if not path.is_file():
            missing.append(task["task_id"])
            rows.append({**task, "path": str(path), "status": "missing", "gate_status": "INCOMPLETE"})
            continue
        try:
            rows.append(
                _validate_checkpoint_parity_report(
                    path, task, protocol, protocol_path, root, immutable, runner_sha
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            invalid.append({"task_id": task["task_id"], "path": str(path), "error": error})
            rows.append({**task, "path": str(path), "status": "invalid", "gate_status": "INCOMPLETE", "error": error})
    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in complete if row["gate_status"] == "FAIL"]
    passed = [row for row in complete if row["gate_status"] == "PASS"]
    if failed:
        gate_status = "FAIL"
    elif len(complete) == len(tasks) and not missing and not invalid:
        gate_status = "PASS"
    elif not complete and len(missing) == len(tasks):
        gate_status = "NOT_YET_RUN"
    else:
        gate_status = "INCOMPLETE"
    return {
        "schema_version": CHECKPOINT_PARITY_SCHEMA,
        "report_dir": str(out_dir),
        "runner": str(CHECKPOINT_PARITY_RUNNER),
        "runner_sha256": runner_sha,
        "protocol_file_sha256": immutable["protocol_file_sha256"],
        "runtime_tree_sha256": immutable["runtime_tree_sha256"],
        "expected_runs": len(tasks),
        "valid_complete_runs": len(complete),
        "passed_runs": len(passed),
        "failed_runs": len(failed),
        "missing_runs": missing,
        "invalid_runs": invalid,
        "gate_status": gate_status,
        "condition": "per run abs(top1_delta_pp) <= 0.05 pp (abs correct-count delta <= 5/10000)",
        "failure_policy": "any verified per-run failure makes final release NO-GO; primary performance evidence is preserved",
        "runs": rows,
    }


def final_release_readiness(
    primary_status: str, checkpoint_parity_status: str
) -> dict[str, Any]:
    blocking_failure = primary_status == "FAIL" or checkpoint_parity_status == "FAIL"
    if blocking_failure:
        status = "NO_GO"
    elif primary_status == "PASS" and checkpoint_parity_status == "PASS":
        status = "READY"
    else:
        status = "INCOMPLETE"
    return {
        "primary_performance_gate_status": primary_status,
        "checkpoint_parity_gate_status": checkpoint_parity_status,
        "required_before_release": True,
        "blocks_campaign_launch": False,
        "blocking_failure_observed": blocking_failure,
        "final_release_status": status,
        "final_release_ready": status == "READY",
    }


def csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in result["accuracy"]["summary"]:
        rows.append(
            {
                "record_type": "accuracy_epoch199_ema",
                "model_id": entry["model_id"],
                "baseline_model": "",
                "resize": entry["resize"],
                "mode": "",
                "metric": entry["metric"],
                "n": entry["n"],
                "mean": entry["mean"],
                "sample_sd": entry["sample_sd"],
                "complete": entry["complete"],
                "seed": "",
                "epoch": 199,
                "value": "",
            }
        )
    for entry in result["accuracy"]["paired_deltas"]:
        rows.append(
            {
                "record_type": "paired_accuracy_delta",
                "model_id": entry["candidate_model"],
                "baseline_model": entry["baseline_model"],
                "resize": entry["resize"],
                "mode": "",
                "metric": entry["metric"],
                "n": entry["n"],
                "mean": entry["mean"],
                "sample_sd": entry["sample_sd"],
                "complete": entry["complete"],
                "seed": "",
                "epoch": 199,
                "value": "",
            }
        )
    for entry in result["efficiency"]["paired_ratios"]:
        for metric in (
            "throughput_ratio",
            "step_time_ratio",
            "peak_allocated_ratio",
            "peak_reserved_ratio",
            "params_ratio",
        ):
            stats = entry[metric]
            rows.append(
                {
                    "record_type": "paired_efficiency_ratio",
                    "model_id": entry["candidate_model"],
                    "baseline_model": entry["baseline_model"],
                    "resize": entry["resize"],
                    "mode": entry["mode"],
                    "metric": metric,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "sample_sd": stats["sample_sd"],
                    "complete": stats["n"] >= result["decision"]["required_gpu_count"] if result["decision"].get("required_gpu_count") else False,
                    "seed": "",
                    "epoch": "",
                    "value": "",
                }
            )
    for entry in result["efficiency"]["parity"]:
        formal = entry["metrics"]["max_abs_diff"]["n"] > 0
        metrics = entry["metrics"] if formal else entry["provisional_metrics"]
        record_type = "parity_formal" if formal else "parity_provisional"
        for metric in ("max_abs_diff", "mean_abs_diff", "argmax_agreement"):
            stats = metrics[metric]
            rows.append(
                {
                    "record_type": record_type,
                    "model_id": entry["model_id"],
                    "baseline_model": "generic_vs_fast",
                    "resize": entry["resize"],
                    "mode": "logits_parity",
                    "metric": metric,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "sample_sd": stats["sample_sd"],
                    "complete": entry["complete"],
                    "seed": "",
                    "epoch": "",
                    "value": "",
                }
            )
        rows.append(
            {
                "record_type": record_type,
                "model_id": entry["model_id"],
                "baseline_model": "generic_vs_fast",
                "resize": entry["resize"],
                "mode": "logits_parity",
                "metric": "allclose_all",
                "n": len(entry["formal_gpus"] if formal else entry["observed_gpus"]),
                "mean": "",
                "sample_sd": "",
                "complete": entry["complete"],
                "seed": "",
                "epoch": "",
                "value": entry["allclose_all"] if formal else entry["provisional_allclose_all"],
            }
        )
    for entry in result["checkpoint_parity"]["runs"]:
        if entry.get("status") != "complete":
            rows.append(
                {
                    "record_type": "checkpoint_parity_run",
                    "model_id": entry["model_id"],
                    "baseline_model": "generic_vs_fast",
                    "resize": entry["resize"],
                    "mode": "epoch199_ema_full_cifar100",
                    "metric": "gate_status",
                    "n": 0,
                    "mean": "",
                    "sample_sd": "",
                    "complete": False,
                    "seed": entry["seed"],
                    "epoch": 199,
                    "value": "INCOMPLETE",
                }
            )
            continue
        for metric in (
            "generic_top1",
            "fast_top1",
            "top1_delta_pp",
            "argmax_agreement",
            "argmax_mismatch_count",
            "max_abs_logit_diff",
            "mean_abs_logit_diff",
        ):
            rows.append(
                {
                    "record_type": "checkpoint_parity_run",
                    "model_id": entry["model_id"],
                    "baseline_model": "generic_vs_fast",
                    "resize": entry["resize"],
                    "mode": "epoch199_ema_full_cifar100",
                    "metric": metric,
                    "n": 10000,
                    "mean": "",
                    "sample_sd": "",
                    "complete": True,
                    "seed": entry["seed"],
                    "epoch": 199,
                    "value": entry[metric],
                }
            )
    # Best is explicitly labeled appendix so downstream users cannot mistake it
    # for the preregistered primary endpoint.
    for entry in result["accuracy"]["runs"]:
        if entry.get("best_top1") is None:
            continue
        rows.append(
            {
                "record_type": "appendix_best_only",
                "model_id": entry["model_id"],
                "baseline_model": "",
                "resize": entry["resize"],
                "mode": "",
                "metric": "best_ema_top1_not_primary",
                "n": 1,
                "mean": "",
                "sample_sd": "",
                "complete": entry["status"] == "complete",
                "seed": entry["seed"],
                "epoch": entry["best_epoch"],
                "value": entry["best_top1"],
            }
        )
    rows.append(
        {
            "record_type": "final_release",
            "model_id": "mn_l4",
            "baseline_model": "deit_s8",
            "resize": "",
            "mode": "primary_plus_checkpoint_parity",
            "metric": "final_release_status",
            "n": result["checkpoint_parity"]["valid_complete_runs"],
            "mean": "",
            "sample_sd": "",
            "complete": result["release_readiness"]["final_release_status"] != "INCOMPLETE",
            "seed": "",
            "epoch": 199,
            "value": result["release_readiness"]["final_release_status"],
        }
    )
    return rows


def render_csv(result: Mapping[str, Any]) -> str:
    import io

    rows = csv_rows(result)
    fields = (
        "record_type",
        "model_id",
        "baseline_model",
        "resize",
        "mode",
        "metric",
        "n",
        "mean",
        "sample_sd",
        "complete",
        "seed",
        "epoch",
        "value",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(result: Mapping[str, Any]) -> str:
    accuracy = result["accuracy"]
    efficiency = result["efficiency"]
    decision = result["decision"]
    checkpoint_parity = result["checkpoint_parity"]
    release = result["release_readiness"]
    lines = [
        "# CIFAR resize 实验聚合",
        "",
        f"生成时间：`{result['generated_at']}`。主指标严格使用 epoch 199 的 EMA top-1；证据不齐全时只显示 `INCOMPLETE`。",
        "",
        f"主实验 λ4 预注册决策：**{decision['status']}**。",
        "",
        f"最终发布状态：**{release['final_release_status']}**（checkpoint 后验：{checkpoint_parity['gate_status']}，{checkpoint_parity['valid_complete_runs']}/{checkpoint_parity['expected_runs']} 份有效证据）。",
        "",
        "主实验 PASS/FAIL 与最终发布状态是两层结论：epoch-199 EMA checkpoint 的完整 CIFAR-100 generic ↔ fast 后验未全部通过前，不能标记 READY；synthetic 初始化态 parity 不能替代该后验。",
        "",
        "## Accuracy：epoch 199 EMA",
        "",
        "| resize | model | complete seeds | top-1 mean ± sample SD | 状态 |",
        "|---:|---|---:|---:|---|",
    ]
    for row in accuracy["summary"]:
        lines.append(
            f"| {row['resize']} | {md_escape(row['model_id'])} | {row['n']}/{row['expected_seeds']} | "
            f"{formatted_stats(row)} | {'COMPLETE' if row['complete'] else 'INCOMPLETE'} |"
        )
    lines.extend(
        [
            "",
            "## 同 seed 配对 accuracy delta",
            "",
            "| resize | candidate − DeiT | paired seeds | delta (pp) | 状态 |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in accuracy["paired_deltas"]:
        lines.append(
            f"| {row['resize']} | {md_escape(row['candidate_model'])} − {md_escape(row['baseline_model'])} | "
            f"{row['n']}/{row['expected_pairs']} | {formatted_stats(row)} | "
            f"{'COMPLETE' if row['complete'] else 'INCOMPLETE'} |"
        )
    lines.extend(
        [
            "",
            "## 8 卡内配对效率比值",
            "",
            "比值均为 candidate / DeiT，先在同一物理卡配对，再跨卡汇总。吞吐越高越好；耗时和显存越低越好。",
            "这些 mandatory train 比值是锁定 batch 的 synthetic、model-only、steady-state step 微基准，不代表 200 epoch 端到端 wall-clock；accuracy run 的吞吐只作后续补充且不进入预注册 gate。",
            "",
            "| resize | mode | candidate | cards | throughput | step time | allocated memory | reserved memory |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in efficiency["paired_ratios"]:
        lines.append(
            f"| {row['resize']} | {row['mode']} | {md_escape(row['candidate_model'])} | "
            f"{row['throughput_ratio']['n']} | {formatted_stats(row['throughput_ratio'], 3)} | "
            f"{formatted_stats(row['step_time_ratio'], 3)} | {formatted_stats(row['peak_allocated_ratio'], 3)} | "
            f"{formatted_stats(row['peak_reserved_ratio'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Synthetic 初始化态 generic ↔ fast logits parity",
            "",
            "完整卡结果优先；尚无完整卡时显示 `provisional`，不作为正式结论。",
            "",
            "| model | resize | cards | allclose | worst max abs | mean abs | argmax agreement | 状态 |",
            "|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in efficiency["parity"]:
        formal = row["metrics"]["max_abs_diff"]["n"] > 0
        metrics = row["metrics"] if formal else row["provisional_metrics"]
        cards = row["formal_gpus"] if formal else row["observed_gpus"]
        allclose = row["allclose_all"] if formal else row["provisional_allclose_all"]
        worst = row["worst_max_abs_diff"] if formal else row["provisional_worst_max_abs_diff"]
        worst_text = "—" if worst is None else f"{worst:.6g}"
        lines.append(
            f"| {md_escape(row['model_id'])} | {row['resize']} | {len(cards)}/{row['expected_gpu_count']} | "
            f"{allclose if allclose is not None else '—'} | {worst_text} | "
            f"{formatted_stats(metrics['mean_abs_diff'], 6)} | "
            f"{formatted_stats(metrics['argmax_agreement'], 4)} | "
            f"{'COMPLETE' if row['complete'] else 'provisional/INCOMPLETE'} |"
        )
    lines.extend(
        [
            "",
            "## Epoch-199 EMA checkpoint 全量 CIFAR-100 parity（最终发布门禁）",
            "",
            "每个 run 对同一 checkpoint、同一批输入依次执行 `alternating_per_layer` 与 `alternating_per_layer_fast`。唯一否决条件是 `|fast top-1 − generic top-1| <= 0.05 pp`（正确数差最多 5/10000）；argmax mismatch 与 logit diff 是诊断项。",
            "",
            "| model | resize | seed | GPU UUID | generic | fast | Δ top-1 (pp) | argmax mismatch | max / mean logit diff | gate |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in checkpoint_parity["runs"]:
        if row.get("status") != "complete":
            lines.append(
                f"| {md_escape(row['model_id'])} | {row['resize']} | {row['seed']} | — | — | — | — | — | — | INCOMPLETE |"
            )
            continue
        lines.append(
            f"| {md_escape(row['model_id'])} | {row['resize']} | {row['seed']} | "
            f"{md_escape(row['gpu_uuid'])} | {row['generic_top1']:.4f} | {row['fast_top1']:.4f} | "
            f"{row['top1_delta_pp']:+.4f} | {row['argmax_mismatch_count']} | "
            f"{row['max_abs_logit_diff']:.6g} / {row['mean_abs_logit_diff']:.6g} | {row['gate_status']} |"
        )
    if checkpoint_parity.get("invalid_runs"):
        lines.extend(["", "非法 parity evidence（fail closed）：", ""])
        for row in checkpoint_parity["invalid_runs"]:
            lines.append(
                f"- `{md_escape(row['task_id'])}`: {md_escape(row['error'])}"
            )
    lines.extend(
        [
            "",
            "## λ4 预注册门槛",
            "",
            "| metric | resize | evidence | rule | value | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for condition in decision.get("conditions", []):
        for row in condition["per_resize"]:
            value = "—" if row["value"] is None else f"{row['value']:.4f}"
            lines.append(
                f"| {condition['metric']} | {row['resize']} | {row['n']}/{row['required_n']} | "
                f"{row['op']} {row['threshold']} | {value} | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "Inference 结果只汇报；当前预注册规则未把 inference 纳入 λ4 PASS。",
            "",
            "## 缺失 / partial runs",
            "",
            "| model | resize | seed | last epoch | status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    partial = [row for row in accuracy["runs"] if row["status"] != "complete"]
    if partial:
        for row in partial:
            lines.append(
                f"| {md_escape(row['model_id'])} | {row['resize']} | {row['seed']} | "
                f"{row.get('last_epoch') if row.get('last_epoch') is not None else '—'} | {row['status']} |"
            )
    else:
        lines.append("| — | — | — | — | none |")
    lines.extend(
        [
            "",
            "## 附录：per-run best（非主指标）",
            "",
            "以下 best 值仅供诊断，不参与主表、paired delta 或 λ4 决策。",
            "",
            "| model | resize | seed | best epoch | best EMA top-1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in accuracy["runs"]:
        if row.get("best_top1") is not None:
            lines.append(
                f"| {md_escape(row['model_id'])} | {row['resize']} | {row['seed']} | "
                f"{row['best_epoch']} | {row['best_top1']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(os.environ.get("PROTOCOL_PATH", DEFAULT_PROTOCOL)),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(os.environ.get("CAMPAIGN_ROOT", SCRIPT_DIR)),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: CAMPAIGN_ROOT/aggregate",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict-complete",
        action="store_true",
        help="exit 2 unless primary 45-job/8-GPU evidence and its gate are resolved; parity is separate",
    )
    strict_group.add_argument(
        "--strict-final-release",
        "--require-release-go",
        dest="strict_final_release",
        action="store_true",
        help="exit 0 only for final READY, 2 for INCOMPLETE, and 3 for conclusive NO_GO",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol.expanduser().resolve()
    root = args.campaign_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else root / "aggregate"
    protocol = load_json(protocol_path)
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol root must be an object")
    models, resizes, seeds = protocol_matrix(protocol)
    runs, accuracy_summary, accuracy_paired, state_info = aggregate_accuracy(protocol, root)
    (
        efficiency_documents,
        efficiency_raw,
        efficiency_paired,
        efficiency_parity,
    ) = aggregate_efficiency(protocol, root)
    decision = evaluate_lambda4_gate(protocol, accuracy_paired, efficiency_paired)
    checkpoint_parity = load_checkpoint_parity(protocol, protocol_path, root)
    release_readiness = final_release_readiness(
        str(decision.get("status", "INCOMPLETE")),
        str(checkpoint_parity.get("gate_status", "INCOMPLETE")),
    )
    accuracy_complete = sum(row["status"] == "complete" for row in runs)
    complete_efficiency_cards = {
        row.get("physical_gpu")
        for row in efficiency_documents
        if row.get("complete") is True and row.get("physical_gpu") is not None
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "protocol": str(protocol_path),
        "campaign_root": str(root),
        "matrix": {
            "models": models,
            "resizes": resizes,
            "seeds": seeds,
            "expected_accuracy_jobs": len(models) * len(resizes) * len(seeds),
        },
        "campaign_state": state_info,
        "completeness": {
            "accuracy_complete_jobs": accuracy_complete,
            "accuracy_expected_jobs": len(runs),
            "efficiency_complete_cards": len(complete_efficiency_cards),
            "efficiency_expected_cards": 8,
        },
        "accuracy": {
            "primary_endpoint": "epoch_199_ema_top1",
            "runs": runs,
            "summary": accuracy_summary,
            "paired_deltas": accuracy_paired,
            "best_values_policy": "appendix_only_not_used_for_decision",
        },
        "efficiency": {
            "documents": efficiency_documents,
            "raw_summary": efficiency_raw,
            "paired_ratios": efficiency_paired,
            "parity": efficiency_parity,
            "pairing_policy": "candidate_over_baseline_within_physical_gpu_then_mean_and_sample_sd",
            "scope": "synthetic_model_only_steady_state_step_not_200_epoch_end_to_end_wall_clock",
        },
        "decision": decision,
        "checkpoint_parity": checkpoint_parity,
        "release_readiness": release_readiness,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "aggregate_results.json", result)
    atomic_write_text(out_dir / "aggregate_results.csv", render_csv(result))
    atomic_write_text(out_dir / "aggregate_results.md", render_markdown(result))
    print(
        json.dumps(
            {
                "json": str(out_dir / "aggregate_results.json"),
                "csv": str(out_dir / "aggregate_results.csv"),
                "markdown": str(out_dir / "aggregate_results.md"),
                "accuracy": f"{accuracy_complete}/{len(runs)}",
                "efficiency_cards": f"{len(complete_efficiency_cards)}/8",
                "lambda4_decision": decision["status"],
                "checkpoint_parity": (
                    f"{checkpoint_parity['valid_complete_runs']}/"
                    f"{checkpoint_parity['expected_runs']}:{checkpoint_parity['gate_status']}"
                ),
                "final_release_status": release_readiness["final_release_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    fully_complete = (
        accuracy_complete == len(runs)
        and len(complete_efficiency_cards) >= 8
        and decision.get("status") in {"PASS", "FAIL"}
    )
    if args.strict_final_release:
        if release_readiness["final_release_status"] == "READY":
            return 0
        if release_readiness["final_release_status"] == "NO_GO":
            return 3
        return 2
    return 0 if fully_complete or not args.strict_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
