from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.cifar_resize_20260810 import build_visual_report as report


ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = ROOT / "reports/evidence/cifar_resize_20260810/aggregate_results.json"
OUTPUT = ROOT / "reports/mergenet_cifar_resize_visual_report_20260814.html"


class VisualReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(AGGREGATE.read_text(encoding="utf-8"))

    def test_render_contains_complete_absolute_train_matrix(self) -> None:
        rendered = report.build_html(self.data, AGGREGATE)
        self.assertEqual(rendered.count("data-absolute-train-efficiency-row="), 15)
        self.assertEqual(
            rendered.count('data-absolute-train-efficiency-row="deit_s8-'), 5
        )
        self.assertEqual(rendered.count("<svg "), 7)
        for expected in (
            "870.2 ± 85.0",
            "2,474.0 ± 0.0",
            "253.6 ± 4.1",
            "8,926.8 ± 0.0",
            "DeiT-S/8 · 归一化基线",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("https://", rendered)
        self.assertNotIn("http://", rendered)

    def test_render_contains_complete_inference_matrix(self) -> None:
        rendered = report.build_html(self.data, AGGREGATE)
        self.assertEqual(rendered.count("data-absolute-inference-row="), 25)
        self.assertEqual(
            rendered.count('data-absolute-inference-row="deit_s8-'), 5
        )
        self.assertNotIn("deit_s8-r160-infer_fast", rendered)
        for expected in (
            "3,869.0 ± 359.5",
            "924.6 ± 5.3",
            "894.9 ± 18.6",
            "943.4 ± 2.8",
            "0.968 ± 0.020×",
            "1.020 ± 0.005×",
            "2.035 ± 0.000×",
            "它仅作补充汇报，不参与预注册训练门禁",
        ):
            self.assertIn(expected, rendered)

    def test_missing_absolute_generic_inference_row_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["efficiency"]["raw_summary"] = [
            row
            for row in mutated["efficiency"]["raw_summary"]
            if not (
                row["mode"] == "infer_generic"
                and row["model_id"] == "deit_s8"
                and row["resize"] == 160
            )
        ]
        with self.assertRaisesRegex(ValueError, "absolute generic inference"):
            report.validate(mutated)

    def test_missing_absolute_fast_inference_row_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["efficiency"]["raw_summary"] = [
            row
            for row in mutated["efficiency"]["raw_summary"]
            if not (
                row["mode"] == "infer_fast"
                and row["model_id"] == "mn_l4"
                and row["resize"] == 320
            )
        ]
        with self.assertRaisesRegex(ValueError, "absolute fast inference"):
            report.validate(mutated)

    def test_invalid_absolute_inference_value_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.data)
        infer_row = next(
            row
            for row in mutated["efficiency"]["raw_summary"]
            if row["mode"] == "infer_fast" and row["model_id"] == "mn_l4"
        )
        infer_row["throughput_img_s"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "absolute fast inference"):
            report.validate(mutated)

    def test_invalid_paired_inference_value_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.data)
        infer_row = next(
            row
            for row in mutated["efficiency"]["paired_ratios"]
            if row["mode"] == "infer_fast" and row["candidate_model"] == "mn_l4"
        )
        infer_row["throughput_ratio"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "paired inference throughput"):
            report.validate(mutated)

    def test_missing_absolute_train_row_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["efficiency"]["raw_summary"] = [
            row
            for row in mutated["efficiency"]["raw_summary"]
            if not (
                row["mode"] == "train"
                and row["model_id"] == "deit_s8"
                and row["resize"] == 160
            )
        ]
        with self.assertRaisesRegex(ValueError, "absolute train efficiency"):
            report.validate(mutated)

    def test_invalid_absolute_train_value_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.data)
        train_row = next(
            row
            for row in mutated["efficiency"]["raw_summary"]
            if row["mode"] == "train"
        )
        train_row["throughput_img_s"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "absolute train efficiency"):
            report.validate(mutated)

    def test_checked_in_report_matches_deterministic_builder(self) -> None:
        self.assertEqual(
            OUTPUT.read_text(encoding="utf-8"),
            report.build_html(self.data, AGGREGATE),
        )

    def test_positive_scaleup_recommendation_preserves_locked_gate(self) -> None:
        rendered = report.build_html(self.data, AGGREGATE)
        self.assertEqual(report.imagenet_scaleup_recommendation(self.data), "GO")
        self.assertEqual(report.gate_score(self.data), (5, 6))
        self.assertEqual(report.condition_score(self.data), (2, 3))
        self.assertIn("ImageNet 推进建议</span><strong><span class=\"status pass\">GO", rendered)
        self.assertIn("建议进入 ImageNet 规模预训练实验", rendered)
        self.assertIn("严格 FAIL/NO_GO 字段保持不变", rendered)
        self.assertIn("逐尺度 5/6 · 顶层 2/3", rendered)
        self.assertIn("FAIL · tracked", rendered)
        self.assertEqual(
            report.gate_rows_html(self.data["decision"]["conditions"]).count("<tr>"),
            6,
        )

    def test_tampered_complete_evidence_is_rejected_by_locked_source(self) -> None:
        mutations = []

        decision_tamper = copy.deepcopy(self.data)
        decision_tamper["decision"]["conditions"][0]["per_resize"][0]["status"] = "FAIL"
        mutations.append(decision_tamper)

        accuracy_tamper = copy.deepcopy(self.data)
        accuracy_tamper["accuracy"]["paired_deltas"][0]["mean"] += 0.25
        mutations.append(accuracy_tamper)

        parity_tamper = copy.deepcopy(self.data)
        parity_tamper["checkpoint_parity"]["gate_status"] = "FAIL"
        mutations.append(parity_tamper)

        for mutated in mutations:
            with self.subTest(mutation=mutations.index(mutated)):
                with self.assertRaisesRegex(ValueError, "unlocked evidence|does not match"):
                    report.build_html(mutated, AGGREGATE)

    def test_parity_failure_holds_scaleup(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["checkpoint_parity"]["gate_status"] = "FAIL"
        mutated["checkpoint_parity"]["failed_runs"] = 1
        mutated["checkpoint_parity"]["runs"][0]["gate_status"] = "FAIL"
        mutated["checkpoint_parity"]["runs"][0]["abs_top1_delta_pp"] = 0.10
        self.assertEqual(report.imagenet_scaleup_recommendation(mutated), "HOLD")
        rendered = report.build_html(mutated, AGGREGATE, _evidence_validated=True)
        self.assertIn("ImageNet 规模预训练暂缓", rendered)
        self.assertIn("checkpoint parity · FAIL", rendered)
        self.assertIn("1 个 checkpoint 未通过", rendered)
        self.assertIn("ImageNet 规模验证暂缓", rendered)
        self.assertNotIn("这些一致趋势支持进入 ImageNet", rendered)
        self.assertNotIn("支持进入 ImageNet 规模验证；", rendered)


if __name__ == "__main__":
    unittest.main()
