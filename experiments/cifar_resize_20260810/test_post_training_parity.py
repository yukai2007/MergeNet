#!/usr/bin/env python3
"""CPU-only tests for the post-training checkpoint parity release gate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity = load_module("cifar_resize_post_training_parity", "post_training_parity.py")
aggregate = load_module("cifar_resize_aggregate_results", "aggregate_results.py")


class ParityRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = HERE / "protocol.json"
        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))

    def test_locked_matrix_has_exactly_30_tasks(self) -> None:
        parity.validate_protocol(self.protocol)
        tasks = parity.build_tasks(self.protocol)
        self.assertEqual(len(tasks), 30)
        self.assertEqual(len({task.task_id for task in tasks}), 30)
        self.assertEqual({task.model_id for task in tasks}, {"mn_l2", "mn_l4"})

    def test_top1_gate_boundary_is_correct_count_delta_not_argmax_mismatch(self) -> None:
        at_boundary = parity.gate_from_counts(7000, 7005, 10_000)
        over_boundary = parity.gate_from_counts(7000, 7006, 10_000)
        negative_boundary = parity.gate_from_counts(7000, 6995, 10_000)
        self.assertTrue(at_boundary["pass"])
        self.assertTrue(negative_boundary["pass"])
        self.assertFalse(over_boundary["pass"])
        self.assertEqual(at_boundary["correct_count_difference"], 5)

    def test_terminal_fail_report_is_reused_and_identity_drift_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            identity = {"task": "mn_l2__r160__s42", "checkpoint_sha256": "abc"}
            parity.atomic_write_json(
                path,
                {
                    "status": "complete",
                    "identity": identity,
                    "gate": {"status": "FAIL", "pass": False},
                },
            )
            reusable, value = parity.reusable_complete_report(path, identity)
            self.assertTrue(reusable)
            self.assertEqual(value["gate"]["status"], "FAIL")
            with self.assertRaisesRegex(ValueError, "identity drift"):
                parity.reusable_complete_report(
                    path, {**identity, "checkpoint_sha256": "changed"}
                )


class AggregateReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = HERE / "protocol.json"
        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))

    def test_primary_and_final_release_statuses_are_separate(self) -> None:
        self.assertEqual(
            aggregate.final_release_readiness("PASS", "PASS")["final_release_status"],
            "READY",
        )
        self.assertEqual(
            aggregate.final_release_readiness("PASS", "INCOMPLETE")["final_release_status"],
            "INCOMPLETE",
        )
        self.assertEqual(
            aggregate.final_release_readiness("INCOMPLETE", "FAIL")["final_release_status"],
            "NO_GO",
        )
        self.assertEqual(
            aggregate.final_release_readiness("FAIL", "NOT_YET_RUN")["final_release_status"],
            "NO_GO",
        )

    def _fixture_report(self, root: Path, correct_delta: int) -> tuple[Path, dict, dict]:
        task = aggregate.checkpoint_parity_tasks(self.protocol)[0]
        job_dir = aggregate.canonical_job_dir(
            root, task["model_id"], task["resize"], task["seed"]
        )
        job_dir.mkdir(parents=True)
        checkpoint_path = job_dir / "last.pth.tar"
        checkpoint_path.write_bytes(b"fixture checkpoint")
        checkpoint_sha = aggregate.sha256_file(checkpoint_path)
        completion_path = job_dir / "completion.json"
        completion = {
            "status": "complete",
            "epoch": 199,
            "ema": True,
            "model_id": task["model_id"],
            "resize": task["resize"],
            "seed": task["seed"],
            "checkpoint_sha256": checkpoint_sha,
        }
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        test_path = root / "data" / "cifar-100-python" / "test"
        test_path.parent.mkdir(parents=True)
        test_path.write_bytes(b"fixture cifar test")

        generic_correct = 7000
        fast_correct = generic_correct + correct_delta
        delta_pp = correct_delta * 100.0 / 10_000
        gate_pass = abs(correct_delta) <= 5
        expected_identity = {
            key: task[key]
            for key in ("task_id", "model_id", "resize", "seed", "validation_batch_size")
        }
        immutable = {
            "runtime_tree_sha256": "runtime-sha",
            "snapshot_manifest_sha256": "snapshot-sha",
            "snapshot_bundle_sha256": "bundle-sha",
        }
        protocol_file_sha = aggregate.sha256_file(self.protocol_path)
        protocol_canonical_sha = aggregate.hashlib.sha256(
            aggregate.canonical_json(self.protocol)
        ).hexdigest()
        runner_sha = aggregate.sha256_file(aggregate.CHECKPOINT_PARITY_RUNNER)
        identity = {
            "runner_revision": 1,
            "runner_sha256": runner_sha,
            "task": expected_identity,
            "protocol_file_sha256": protocol_file_sha,
            "protocol_canonical_sha256": protocol_canonical_sha,
            "runtime_tree_sha256": immutable["runtime_tree_sha256"],
            "checkpoint_sha256": checkpoint_sha,
        }
        report = {
            "schema_version": aggregate.CHECKPOINT_PARITY_SCHEMA,
            "runner_revision": 1,
            "status": "complete",
            "identity": identity,
            "task": expected_identity,
            "environment": {
                "canonical": True,
                "versions": aggregate.EXPECTED_RELEASE_ENVIRONMENT,
                "runtime_env": aggregate.EXPECTED_RELEASE_RUNTIME_ENV,
                "gpu": {"physical_index": 7, "uuid": "GPU-fixture"},
            },
            "runtime": {
                "tree_sha256": immutable["runtime_tree_sha256"],
                "snapshot_manifest_sha256": immutable["snapshot_manifest_sha256"],
                "snapshot_bundle_sha256": immutable["snapshot_bundle_sha256"],
            },
            "checkpoint": {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "completion_sha256": aggregate.sha256_file(completion_path),
                "checkpoint_epoch": 199,
                "ema_state_key": "state_dict_ema",
                "strict_state_dict_load": True,
            },
            "data": {
                "test_path": str(test_path.resolve()),
                "test_md5": "f0ef6b0ae62326f3e7ffdfab6717acfc",
                "test_sha256": aggregate.sha256_file(test_path),
                "expected_samples": 10_000,
            },
            "evaluation": {
                "dataset": "CIFAR100",
                "split": "test",
                "sample_count": 10_000,
                "class_count": 100,
                "loader_deterministic": True,
                "loader_shared_between_modes": True,
                "validation_batch_size": task["validation_batch_size"],
                "amp": True,
                "amp_dtype": "float16",
                "generic_grouping": "alternating_per_layer",
                "fast_grouping": "alternating_per_layer_fast",
                "grouping_seed": 0,
                "generic_correct": generic_correct,
                "fast_correct": fast_correct,
                "generic_top1": generic_correct / 100,
                "fast_top1": fast_correct / 100,
                "top1_delta_pp": delta_pp,
                "abs_top1_delta_pp": abs(delta_pp),
                "argmax_agreement_count": 9980,
                "argmax_mismatch_count": 20,
                "argmax_agreement": 0.998,
                "max_abs_logit_diff": 0.01,
                "mean_abs_logit_diff": 0.001,
                "fast_vs_training_summary_delta_pp": 0.0,
            },
            "gate": {
                "status": "PASS" if gate_pass else "FAIL",
                "pass": gate_pass,
                "threshold_pp": 0.05,
                "max_correct_count_difference": 5,
                "correct_count_difference": correct_delta,
            },
        }
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, task, immutable

    def test_aggregator_recomputes_pass_and_fail_boundary(self) -> None:
        for delta, expected in ((5, "PASS"), (6, "FAIL")):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                report_path, task, immutable = self._fixture_report(root, delta)
                row = aggregate._validate_checkpoint_parity_report(
                    report_path,
                    task,
                    self.protocol,
                    self.protocol_path,
                    root,
                    immutable,
                    aggregate.sha256_file(aggregate.CHECKPOINT_PARITY_RUNNER),
                )
                self.assertEqual(row["gate_status"], expected)

    def test_aggregator_rejects_self_reported_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path, task, immutable = self._fixture_report(root, 5)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["evaluation"]["top1_delta_pp"] = 0.0
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top1_delta_pp"):
                aggregate._validate_checkpoint_parity_report(
                    report_path,
                    task,
                    self.protocol,
                    self.protocol_path,
                    root,
                    immutable,
                    aggregate.sha256_file(aggregate.CHECKPOINT_PARITY_RUNNER),
                )


if __name__ == "__main__":
    unittest.main()
