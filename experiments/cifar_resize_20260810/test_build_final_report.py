#!/usr/bin/env python3
"""CPU-only fixtures for the fail-closed CIFAR final-report publisher."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_report as builder


def stats(values: list[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def empty_stats() -> dict[str, object]:
    return {"n": 0, "mean": None, "sample_sd": None, "values": []}


def top1(model: str, resize: int, seed: int) -> float:
    baseline = 62.0 + (resize - 160) * 0.01 + (seed - 43) * 0.1
    return baseline + {"deit_s8": 0.0, "mn_l2": 2.5, "mn_l4": 1.0}[model]


def make_decision(ready: bool) -> dict[str, object]:
    specification = (
        ("paired_accuracy_delta_pp", ">", 0.0, {256: 1.0, 320: 1.0}, 3),
        ("train_throughput_ratio", ">=", 1.0, {256: 1.05 if ready else 0.95, 320: 1.08}, 8),
        ("train_peak_allocated_ratio", "<", 1.0, {256: 0.82, 320: 0.80}, 8),
    )
    conditions = []
    for metric, op, threshold, values, n in specification:
        per_resize = []
        for resize in (256, 320):
            value = values[resize]
            passed = builder.evaluate_rule(value, op, threshold)
            per_resize.append(
                {
                    "resize": resize,
                    "value": value,
                    "sample_sd": 0.01,
                    "n": n,
                    "required_n": n,
                    "op": op,
                    "threshold": threshold,
                    "status": "PASS" if passed else "FAIL",
                }
            )
        conditions.append(
            {
                "metric": metric,
                "status": "FAIL" if any(row["status"] == "FAIL" for row in per_resize) else "PASS",
                "per_resize": per_resize,
            }
        )
    return {
        "status": "FAIL" if any(row["status"] == "FAIL" for row in conditions) else "PASS",
        "candidate_model": "mn_l4",
        "baseline_model": "deit_s8",
        "required_complete_seeds": 3,
        "required_gpu_count": 8,
        "conditions": conditions,
    }


def make_document(ready: bool = False, parity_failure: bool = False) -> dict[str, object]:
    accuracy_runs = []
    summaries = []
    paired = []
    for resize in builder.RESIZES:
        for model in builder.MODELS:
            values = []
            for seed in builder.SEEDS:
                value = top1(model, resize, seed)
                values.append(value)
                accuracy_runs.append(
                    {
                        "model_id": model,
                        "resize": resize,
                        "seed": seed,
                        "target_epoch": 199,
                        "last_epoch": 199,
                        "ema_top1": value,
                        "best_top1": value + 0.1,
                        "best_epoch": 180,
                        "artifact_verified": True,
                        "status": "complete",
                    }
                )
            summaries.append(
                {
                    "model_id": model,
                    "resize": resize,
                    "metric": "epoch_199_ema_top1",
                    "expected_seeds": 3,
                    "complete_seeds": list(builder.SEEDS),
                    "missing_or_unverified_seeds": [],
                    "complete": True,
                    **stats(values),
                }
            )
        for model in builder.MERGENET_MODELS:
            pairs = []
            deltas = []
            for seed in builder.SEEDS:
                candidate = top1(model, resize, seed)
                baseline = top1("deit_s8", resize, seed)
                delta = candidate - baseline
                deltas.append(delta)
                pairs.append(
                    {
                        "seed": seed,
                        "candidate_top1": candidate,
                        "baseline_top1": baseline,
                        "delta_pp": delta,
                    }
                )
            paired.append(
                {
                    "candidate_model": model,
                    "baseline_model": "deit_s8",
                    "resize": resize,
                    "metric": "paired_accuracy_delta_pp",
                    "expected_pairs": 3,
                    "complete": True,
                    "pairs": pairs,
                    **stats(deltas),
                }
            )

    documents = [
        {"path": f"/campaign/efficiency/gpu{gpu}.json", "physical_gpu": gpu, "complete": True, "item_count": 50}
        for gpu in builder.GPUS
    ]
    raw_summary = []
    for resize in builder.RESIZES:
        for mode in builder.MODE_LABELS:
            for model in builder.MODELS:
                meaningful = not (model == "deit_s8" and mode == "infer_fast")
                base = 1000.0 + resize + builder.MODELS.index(model) * 40
                metric_values = [base + gpu for gpu in builder.GPUS]
                raw_summary.append(
                    {
                        "model_id": model,
                        "resize": resize,
                        "mode": mode,
                        "physical_gpus": list(builder.GPUS) if meaningful else [],
                        "complete_physical_gpus": list(builder.GPUS) if meaningful else [],
                        "provisional": False,
                        "throughput_img_s": stats(metric_values) if meaningful else empty_stats(),
                        "step_time_ms": stats([100000 / value for value in metric_values]) if meaningful else empty_stats(),
                        "peak_allocated_mib": stats([2000 + resize + builder.MODELS.index(model) * 100.0] * 8) if meaningful else empty_stats(),
                        "peak_reserved_mib": stats([2200 + resize + builder.MODELS.index(model) * 100.0] * 8) if meaningful else empty_stats(),
                        "params": stats([21_000_000 + builder.MODELS.index(model) * 500_000.0] * 8) if meaningful else empty_stats(),
                    }
                )
    paired_ratios = []
    for resize in builder.RESIZES:
        for mode in builder.MODE_LABELS:
            for model in builder.MERGENET_MODELS:
                if model == "mn_l4" and mode == "train" and resize in {256, 320}:
                    target = (1.05 if ready else 0.95) if resize == 256 else 1.08
                else:
                    target = 0.7 + builder.MERGENET_MODELS.index(model) * 0.2 + (resize - 160) / 1000
                values = [target + (gpu - 3.5) / 1000 for gpu in builder.GPUS]
                if model == "mn_l4" and mode == "train" and resize in {256, 320}:
                    allocated_target = 0.82 if resize == 256 else 0.80
                else:
                    allocated_target = 0.8035
                paired_ratios.append(
                    {
                        "candidate_model": model,
                        "baseline_model": "deit_s8",
                        "resize": resize,
                        "mode": mode,
                        "physical_gpus": list(builder.GPUS),
                        "throughput_ratio": stats(values),
                        "step_time_ratio": stats([1 / value for value in values]),
                        "peak_allocated_ratio": stats([allocated_target + (gpu - 3.5) / 1000 for gpu in builder.GPUS]),
                        "peak_reserved_ratio": stats([0.9 + gpu / 1000 for gpu in builder.GPUS]),
                        "params_ratio": stats([1.04] * 8),
                    }
                )
    synthetic_parity = []
    for resize in builder.RESIZES:
        for model in builder.MERGENET_MODELS:
            synthetic_parity.append(
                {
                    "model_id": model,
                    "resize": resize,
                    "expected_gpu_count": 8,
                    "observed_gpus": list(builder.GPUS),
                    "formal_gpus": list(builder.GPUS),
                    "complete": True,
                    "allclose_all": True,
                    "worst_max_abs_diff": 0.001,
                    "metrics": {
                        "max_abs_diff": stats([0.001] * 8),
                        "mean_abs_diff": stats([0.0001] * 8),
                        "argmax_agreement": stats([1.0] * 8),
                    },
                }
            )

    parity_runs = []
    passed = 0
    failed = 0
    first = True
    for resize in builder.RESIZES:
        for model in builder.MERGENET_MODELS:
            for seed in builder.SEEDS:
                generic = 7000
                delta_count = 6 if parity_failure and first else 1
                first = False
                fast = generic + delta_count
                mismatch = 10
                gate = "PASS" if abs(delta_count) <= 5 else "FAIL"
                passed += gate == "PASS"
                failed += gate == "FAIL"
                parity_runs.append(
                    {
                        "task_id": f"{model}__r{resize}__s{seed}",
                        "model_id": model,
                        "resize": resize,
                        "seed": seed,
                        "validation_batch_size": 32,
                        "status": "complete",
                        "gate_status": gate,
                        "gate_pass": gate == "PASS",
                        "gpu_index": (seed - 42) % 8,
                        "gpu_uuid": f"GPU-fixture-{(seed - 42) % 8}",
                        "generic_correct": generic,
                        "fast_correct": fast,
                        "generic_top1": generic / 100,
                        "fast_top1": fast / 100,
                        "top1_delta_pp": delta_count / 100,
                        "abs_top1_delta_pp": abs(delta_count) / 100,
                        "argmax_agreement_count": 10000 - mismatch,
                        "argmax_mismatch_count": mismatch,
                        "argmax_agreement": (10000 - mismatch) / 10000,
                        "max_abs_logit_diff": 0.002,
                        "mean_abs_logit_diff": 0.0002,
                    }
                )
    decision = make_decision(ready)
    parity_gate = "PASS" if failed == 0 else "FAIL"
    release = "READY" if decision["status"] == "PASS" and parity_gate == "PASS" else "NO_GO"
    return {
        "schema_version": 2,
        "generated_at": "2026-08-14T12:00:00+00:00",
        "protocol": "/campaign/runtime/protocol.json",
        "campaign_root": "/campaign",
        "matrix": {
            "models": list(builder.MODELS),
            "resizes": list(builder.RESIZES),
            "seeds": list(builder.SEEDS),
            "expected_accuracy_jobs": 45,
        },
        "completeness": {
            "accuracy_complete_jobs": 45,
            "accuracy_expected_jobs": 45,
            "efficiency_complete_cards": 8,
            "efficiency_expected_cards": 8,
        },
        "accuracy": {
            "primary_endpoint": "epoch_199_ema_top1",
            "runs": accuracy_runs,
            "summary": summaries,
            "paired_deltas": paired,
        },
        "efficiency": {
            "documents": documents,
            "raw_summary": raw_summary,
            "paired_ratios": paired_ratios,
            "parity": synthetic_parity,
        },
        "decision": decision,
        "checkpoint_parity": {
            "expected_runs": 30,
            "valid_complete_runs": 30,
            "passed_runs": passed,
            "failed_runs": failed,
            "missing_runs": [],
            "invalid_runs": [],
            "gate_status": parity_gate,
            "runs": parity_runs,
        },
        "release_readiness": {
            "primary_performance_gate_status": decision["status"],
            "checkpoint_parity_gate_status": parity_gate,
            "required_before_release": True,
            "blocking_failure_observed": release == "NO_GO",
            "final_release_status": release,
            "final_release_ready": release == "READY",
        },
    }


def write_bundle(path: Path, document: dict[str, object]) -> dict[str, bytes]:
    path.mkdir(parents=True)
    csv_bytes, markdown_bytes = builder.canonical_projection_bytes(document)
    values = {
        "aggregate_results.json": (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        "aggregate_results.csv": csv_bytes,
        "aggregate_results.md": markdown_bytes,
    }
    for name, value in values.items():
        (path / name).write_bytes(value)
    return values


def make_repo(path: Path) -> None:
    (path / "reports/evidence").mkdir(parents=True)
    (path / "deliverables/imagenet_longtrain_v1/configs").mkdir(parents=True)
    (path / "README.md").write_text(
        "# Fixture release\n\n" + builder.ROOT_CAMPAIGN_LEGACY + "\n",
        encoding="utf-8",
    )
    (path / builder.INDEX_DOC).write_text("# Evidence index\n", encoding="utf-8")
    (path / builder.POSITION_DOC).write_text("# λ4 positioning\n", encoding="utf-8")
    (path / builder.HANDOFF_README).write_text(
        "# ImageNet handoff\n\n"
        "- `configs/mergenet_lambda4.yaml` is the recommended long-run candidate: p8, 4 local + 8 latent blocks, lambda=4, local window 32, and the deterministic fast evaluation grouping.\n\n"
        "The architecture and efficiency choices above come from the completed CIFAR-100 campaign. **ImageNet-1K accuracy has not yet been measured.** Treat lambda=4 as the recommended scale-up candidate, not as a claimed ImageNet result. Run the baseline and lambda=4 under the same protocol; use lambda=2 if the lambda=4 accuracy curve is clearly under the baseline early in training.\n",
        encoding="utf-8",
    )
    (path / builder.HANDOFF_NOTES).write_text(
        "# Release notes\n\n"
        "- 主训练：`configs/mergenet_lambda4.yaml`，4 local + 8 latent、lambda=4、window 32；这是 CIFAR-100 工程结果支持的效率优先候选。\n",
        encoding="utf-8",
    )
    (path / builder.LAMBDA4_CONFIG).write_text(
        "# Recommended ImageNet scale-up candidate, transferred from the CIFAR-100\n"
        "# efficiency winner. ImageNet accuracy is not yet established.\n"
        "model: mergenet_small_cls\nlambda_local: 4.0\n",
        encoding="utf-8",
    )


def tree_snapshot(path: Path) -> list[tuple[str, str, bytes | None]]:
    result: list[tuple[str, str, bytes | None]] = []
    for entry in sorted(path.rglob("*")):
        relative = str(entry.relative_to(path))
        if entry.is_dir():
            result.append((relative, "directory", None))
        elif entry.is_file():
            result.append((relative, "file", entry.read_bytes()))
        else:
            result.append((relative, "other", None))
    return result


class FinalReportBuilderTest(unittest.TestCase):
    def run_main(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = builder.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_incomplete_aggregate_fails_without_repo_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            before = {path.relative_to(repo): path.read_bytes() for path in repo.rglob("*") if path.is_file()}
            document = make_document()
            document["completeness"]["accuracy_complete_jobs"] = 44
            write_bundle(aggregate, document)
            code, _, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo)])
            self.assertEqual(code, 2)
            self.assertIn("must equal 45", stderr)
            after = {path.relative_to(repo): path.read_bytes() for path in repo.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_missing_parity_and_csv_drift_fail_closed(self) -> None:
        for mutation, expected in (
            ("missing", "matrix mismatch"),
            ("csv", "not the byte-exact canonical projection"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo = root / "repo"
                aggregate = root / "aggregate"
                make_repo(repo)
                document = make_document()
                values = write_bundle(aggregate, document)
                if mutation == "missing":
                    document["checkpoint_parity"]["runs"].pop()
                    (aggregate / "aggregate_results.json").write_text(json.dumps(document), encoding="utf-8")
                else:
                    text = values["aggregate_results.csv"].decode("utf-8").replace(",62.0,", ",99.0,", 1)
                    (aggregate / "aggregate_results.csv").write_text(text, encoding="utf-8")
                code, _, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo)])
                self.assertEqual(code, 2)
                self.assertIn(expected, stderr)
                self.assertFalse((repo / builder.FINAL_HTML).exists())

    def test_canonical_csv_and_markdown_numeric_mutations_fail(self) -> None:
        for mutation in ("csv_efficiency", "markdown_efficiency", "markdown_decision"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo = root / "repo"
                aggregate = root / "aggregate"
                make_repo(repo)
                document = make_document(ready=False)
                write_bundle(aggregate, document)
                if mutation == "csv_efficiency":
                    csv_path = aggregate / "aggregate_results.csv"
                    with csv_path.open("r", encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        fieldnames = reader.fieldnames
                        rows = list(reader)
                    target = next(
                        row for row in rows
                        if row["record_type"] == "paired_efficiency_ratio"
                        and row["model_id"] == "mn_l4"
                        and row["resize"] == "256"
                        and row["mode"] == "train"
                        and row["metric"] == "throughput_ratio"
                    )
                    target["mean"] = "123.456"
                    with csv_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(rows)
                elif mutation == "markdown_efficiency":
                    markdown_path = aggregate / "aggregate_results.md"
                    text = markdown_path.read_text(encoding="utf-8")
                    source = next(
                        row for row in document["efficiency"]["paired_ratios"]
                        if row["candidate_model"] == "mn_l4"
                        and row["resize"] == 256
                        and row["mode"] == "train"
                    )
                    old = (
                        f"| 256 | train | mn_l4 | 8 | "
                        f"{source['throughput_ratio']['mean']:.3f} ± {source['throughput_ratio']['sample_sd']:.3f}"
                    )
                    self.assertEqual(text.count(old), 1)
                    markdown_path.write_text(text.replace(old, old.replace("0.950", "0.951", 1)), encoding="utf-8")
                else:
                    markdown_path = aggregate / "aggregate_results.md"
                    text = markdown_path.read_text(encoding="utf-8")
                    old = "| train_throughput_ratio | 256 | 8/8 | >= 1.0 | 0.9500 | FAIL |"
                    self.assertEqual(text.count(old), 1)
                    markdown_path.write_text(text.replace(old, old.replace("0.9500", "0.9501")), encoding="utf-8")
                code, _, stderr = self.run_main(
                    ["--aggregate-dir", str(aggregate), "--repo-root", str(repo), "--check-only"]
                )
                self.assertEqual(code, 2)
                self.assertIn("not the byte-exact canonical projection", stderr)
                self.assertFalse((repo / builder.FINAL_HTML).exists())

    def test_renderer_sha_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            write_bundle(aggregate, make_document())
            with mock.patch.object(builder, "PINNED_AGGREGATE_RENDERER_SHA256", "0" * 64):
                code, _, stderr = self.run_main(
                    ["--aggregate-dir", str(aggregate), "--repo-root", str(repo), "--check-only"]
                )
            self.assertEqual(code, 2)
            self.assertIn("renderer SHA-256 drift", stderr)
            self.assertFalse((repo / builder.FINAL_HTML).exists())

    def test_decision_rule_drift_cannot_turn_no_go_into_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            document = make_document(ready=False)
            throughput = next(
                row for row in document["decision"]["conditions"]
                if row["metric"] == "train_throughput_ratio"
            )
            for row in throughput["per_resize"]:
                row["threshold"] = 0.0
                row["status"] = "PASS"
            throughput["status"] = "PASS"
            document["decision"]["status"] = "PASS"
            document["release_readiness"].update(
                primary_performance_gate_status="PASS",
                blocking_failure_observed=False,
                final_release_status="READY",
                final_release_ready=True,
            )
            write_bundle(aggregate, document)
            code, _, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo), "--check-only"])
            self.assertEqual(code, 2)
            self.assertIn("rule drift", stderr)

    def test_ready_check_only_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            write_bundle(aggregate, make_document(ready=True))
            before = {path.relative_to(repo): path.read_bytes() for path in repo.rglob("*") if path.is_file()}
            code, stdout, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo), "--check-only"])
            self.assertEqual((code, stderr), (0, ""))
            payload = json.loads(stdout)
            self.assertEqual(payload["final_release_status"], "READY")
            self.assertFalse(payload["published"])
            after = {path.relative_to(repo): path.read_bytes() for path in repo.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_conclusive_no_go_publishes_complete_report_atomically_and_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            source = write_bundle(aggregate, make_document(ready=False))
            code, stdout, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo)])
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(json.loads(stdout)["final_release_status"], "NO_GO")
            report = (repo / builder.FINAL_HTML).read_text(encoding="utf-8")
            self.assertIn("最终状态 <span class=\"badge fail\">NO_GO", report)
            self.assertEqual(report.count("data-accuracy-row="), 15)
            self.assertEqual(report.count("data-paired-accuracy-row="), 10)
            self.assertEqual(report.count("data-efficiency-raw-row="), 40)
            self.assertEqual(report.count("data-efficiency-paired-row="), 30)
            self.assertEqual(report.count("data-synthetic-parity-row="), 10)
            self.assertEqual(report.count("data-checkpoint-parity-row="), 30)
            for name, value in source.items():
                self.assertEqual((repo / builder.EVIDENCE_DIR / name).read_bytes(), value)
            manifest = json.loads((repo / builder.EVIDENCE_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["final_release_status"], "NO_GO")
            for name, value in source.items():
                self.assertEqual(manifest["files"][name]["sha256"], hashlib.sha256(value).hexdigest())
            for relative in (builder.ROOT_README, builder.INDEX_DOC, builder.POSITION_DOC):
                content = (repo / relative).read_text(encoding="utf-8")
                self.assertEqual(content.count(builder.MARKER_START), 1)
                self.assertEqual(content.count(builder.MARKER_END), 1)
            root_readme = (repo / builder.ROOT_README).read_text(encoding="utf-8")
            self.assertNotIn("currently running", root_readme)
            self.assertIn("completed 45-run accuracy / 8-card efficiency / 30-checkpoint parity", root_readme)
            handoff_readme = (repo / builder.HANDOFF_README).read_text(encoding="utf-8")
            handoff_notes = (repo / builder.HANDOFF_NOTES).read_text(encoding="utf-8")
            config = (repo / builder.LAMBDA4_CONFIG).read_text(encoding="utf-8")
            self.assertNotIn("recommended long-run candidate", handoff_readme)
            self.assertNotIn("recommended scale-up candidate", handoff_readme)
            self.assertNotIn("效率优先候选", handoff_notes)
            self.assertNotIn("efficiency winner", config)
            self.assertIn("runnable exploratory", handoff_readme)
            self.assertIn("exploratory 候选", handoff_notes)
            self.assertIn("Runnable exploratory ImageNet candidate", config)
            self.assertEqual(config.count("model: mergenet_small_cls"), 1)
            self.assertEqual(config.count("lambda_local: 4.0"), 1)
            before = {path.relative_to(repo): hashlib.sha256(path.read_bytes()).hexdigest() for path in repo.rglob("*") if path.is_file()}
            second_code, _, second_stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo)])
            self.assertEqual((second_code, second_stderr), (0, ""))
            after = {path.relative_to(repo): hashlib.sha256(path.read_bytes()).hexdigest() for path in repo.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_replace_failure_rolls_back_existing_new_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            write_bundle(aggregate, make_document(ready=False))
            before = tree_snapshot(repo)
            self.assertFalse((repo / builder.EVIDENCE_DIR).exists())
            self.assertFalse((repo / builder.FINAL_HTML).exists())

            real_replace = builder.os.replace
            calls = 0

            def fail_once_during_install(source: object, destination: object) -> None:
                nonlocal calls
                calls += 1
                # Sorted publication order has already replaced/restored-file
                # candidates and installed files under the newly created
                # evidence directory by this point.
                if calls == 13:
                    raise OSError("injected os.replace failure")
                real_replace(source, destination)

            with mock.patch.object(builder.os, "replace", side_effect=fail_once_during_install):
                code, _, stderr = self.run_main(
                    ["--aggregate-dir", str(aggregate), "--repo-root", str(repo)]
                )
            self.assertEqual(code, 1)
            self.assertGreaterEqual(calls, 13)
            self.assertIn("injected os.replace failure", stderr)
            self.assertEqual(tree_snapshot(repo), before)
            self.assertFalse((repo / builder.EVIDENCE_DIR).exists())
            self.assertFalse((repo / builder.FINAL_HTML).exists())
            self.assertFalse(any(path.name.startswith(".cifar-final-") for path in repo.iterdir()))

    def test_parity_failure_is_publishable_no_go_not_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            document = make_document(ready=True, parity_failure=True)
            write_bundle(aggregate, document)
            code, stdout, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo), "--check-only"])
            self.assertEqual((code, stderr), (0, ""))
            payload = json.loads(stdout)
            self.assertEqual(payload["checkpoint_parity"], "30/30:FAIL")
            self.assertEqual(payload["final_release_status"], "NO_GO")

    def test_ready_handoff_wording_is_gate_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            aggregate = root / "aggregate"
            make_repo(repo)
            write_bundle(aggregate, make_document(ready=True))
            code, _, stderr = self.run_main(["--aggregate-dir", str(aggregate), "--repo-root", str(repo)])
            self.assertEqual((code, stderr), (0, ""))
            readme = (repo / builder.HANDOFF_README).read_text(encoding="utf-8")
            notes = (repo / builder.HANDOFF_NOTES).read_text(encoding="utf-8")
            config = (repo / builder.LAMBDA4_CONFIG).read_text(encoding="utf-8")
            self.assertIn("gate-qualified ImageNet scale-up candidate", readme)
            self.assertIn("ImageNet accuracy and efficiency remain unmeasured", readme)
            self.assertIn("不代表 ImageNet 已验证", notes)
            self.assertIn("not an established ImageNet result", config)
            self.assertNotIn("efficiency winner", config)

    def test_all_performance_parity_status_combinations_have_accurate_handoff_copy(self) -> None:
        cases = (
            (True, False, "PASS", "PASS", "READY"),
            (False, False, "FAIL", "PASS", "NO_GO"),
            (True, True, "PASS", "FAIL", "NO_GO"),
            (False, True, "FAIL", "FAIL", "NO_GO"),
        )
        for performance_pass, parity_failure, decision, parity, release in cases:
            with self.subTest(decision=decision, parity=parity), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo = root / "repo"
                aggregate = root / "aggregate"
                make_repo(repo)
                document = make_document(ready=performance_pass, parity_failure=parity_failure)
                write_bundle(aggregate, document)
                code, _, stderr = self.run_main(
                    ["--aggregate-dir", str(aggregate), "--repo-root", str(repo)]
                )
                self.assertEqual((code, stderr), (0, ""))
                readme = (repo / builder.HANDOFF_README).read_text(encoding="utf-8")
                notes = (repo / builder.HANDOFF_NOTES).read_text(encoding="utf-8")
                config = (repo / builder.LAMBDA4_CONFIG).read_text(encoding="utf-8")
                root_readme = (repo / builder.ROOT_README).read_text(encoding="utf-8")
                combined = "\n".join((readme, notes, config))
                self.assertIn(
                    f"primary performance gate: {decision}; checkpoint parity: {parity}; release: {release}",
                    config,
                )
                self.assertNotIn("currently running", root_readme)
                if performance_pass:
                    self.assertIn("performance-gate-qualified", combined)
                    self.assertNotIn("cross-scale primary performance gate was not passed", combined)
                    self.assertNotIn("跨尺度性能门禁未通过", combined)
                else:
                    self.assertIn("cross-scale primary performance gate was not passed", combined)
                    self.assertIn("跨尺度性能门禁未通过", combined)
                    self.assertNotIn("lambda4 is the performance-gate-qualified", config)
                if parity_failure:
                    self.assertIn("checkpoint parity failed", combined)
                    self.assertIn("后验失败", combined)
                    self.assertIn("final release is no_go", combined.lower())
                else:
                    self.assertIn("parity passed 30/30", combined)
                    self.assertNotIn("checkpoint parity failed", combined)


if __name__ == "__main__":
    unittest.main()
