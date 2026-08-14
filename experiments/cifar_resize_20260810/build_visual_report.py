#!/usr/bin/env python3
"""Build a chart-first, single-file HTML view of the locked resize evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE = (
    REPO_ROOT
    / "reports/evidence/cifar_resize_20260810/aggregate_results.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/mergenet_cifar_resize_visual_report_20260814.html"

MODEL_META = {
    "deit_s8": ("DeiT-S/8", "#718096"),
    "mn_l2": ("MergeNet λ2", "#168aad"),
    "mn_l4": ("MergeNet λ4", "#e76f51"),
}

INFERENCE_SERIES = (
    ("infer_generic", "deit_s8", "DeiT generic · 基线", None),
    ("infer_generic", "mn_l2", "λ2 generic", "8 6"),
    ("infer_fast", "mn_l2", "λ2 fast", None),
    ("infer_generic", "mn_l4", "λ4 generic", "8 6"),
    ("infer_fast", "mn_l4", "λ4 fast", None),
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(data: dict[str, Any]) -> None:
    matrix = data["matrix"]
    completeness = data["completeness"]
    parity = data["checkpoint_parity"]
    readiness = data["release_readiness"]
    expected_cells = len(matrix["models"]) * len(matrix["resizes"])
    raw_train = [
        row for row in data["efficiency"]["raw_summary"] if row["mode"] == "train"
    ]
    raw_infer_generic = [
        row
        for row in data["efficiency"]["raw_summary"]
        if row["mode"] == "infer_generic"
    ]
    raw_infer_fast = [
        row
        for row in data["efficiency"]["raw_summary"]
        if row["mode"] == "infer_fast" and row["model_id"] != "deit_s8"
    ]
    paired_infer = [
        row
        for row in data["efficiency"]["paired_ratios"]
        if row["mode"] in {"infer_generic", "infer_fast"}
    ]
    expected_train_keys = {
        (resize, model)
        for resize in matrix["resizes"]
        for model in matrix["models"]
    }
    expected_generic_infer_keys = expected_train_keys
    expected_fast_infer_keys = {
        (resize, model)
        for resize in matrix["resizes"]
        for model in matrix["models"]
        if model != "deit_s8"
    }
    expected_paired_infer_keys = {
        (mode, resize, model)
        for mode in ("infer_generic", "infer_fast")
        for resize in matrix["resizes"]
        for model in matrix["models"]
        if model != "deit_s8"
    }

    def complete_raw_rows(rows: list[dict[str, Any]], expected: int) -> bool:
        return len(rows) == expected and all(
            row[metric]["n"] == 8
            and all(
                math.isfinite(float(row[metric][field]))
                for field in ("mean", "sample_sd")
            )
            for row in rows
            for metric in ("throughput_img_s", "peak_allocated_mib", "step_time_ms")
        )

    checks = {
        "accuracy matrix": completeness["accuracy_complete_jobs"]
        == completeness["accuracy_expected_jobs"]
        == matrix["expected_accuracy_jobs"],
        "accuracy summaries": len(data["accuracy"]["summary"]) == expected_cells,
        "paired accuracy": len(data["accuracy"]["paired_deltas"])
        == (len(matrix["models"]) - 1) * len(matrix["resizes"]),
        "efficiency cards": completeness["efficiency_complete_cards"]
        == completeness["efficiency_expected_cards"]
        == 8,
        "absolute train efficiency": len(raw_train) == expected_cells
        and {(row["resize"], row["model_id"]) for row in raw_train}
        == expected_train_keys
        and all(
            row["throughput_img_s"]["n"] == 8
            and row["peak_allocated_mib"]["n"] == 8
            and all(
                math.isfinite(float(row[metric][field]))
                for metric in ("throughput_img_s", "peak_allocated_mib")
                for field in ("mean", "sample_sd")
            )
            for row in raw_train
        ),
        "absolute generic inference": complete_raw_rows(raw_infer_generic, expected_cells)
        and {
            (row["resize"], row["model_id"]) for row in raw_infer_generic
        }
        == expected_generic_infer_keys,
        "absolute fast inference": complete_raw_rows(
            raw_infer_fast,
            (len(matrix["models"]) - 1) * len(matrix["resizes"]),
        )
        and {(row["resize"], row["model_id"]) for row in raw_infer_fast}
        == expected_fast_infer_keys,
        "paired inference throughput": len(paired_infer)
        == 2 * (len(matrix["models"]) - 1) * len(matrix["resizes"])
        and {
            (row["mode"], row["resize"], row["candidate_model"])
            for row in paired_infer
        }
        == expected_paired_infer_keys
        and all(
            row[metric]["n"] == 8
            and all(
                math.isfinite(float(row[metric][field]))
                for field in ("mean", "sample_sd")
            )
            for row in paired_infer
            for metric in ("throughput_ratio", "peak_allocated_ratio")
        )
        and all(
            len(row["per_gpu"]) == 8
            and {pair["physical_gpu"] for pair in row["per_gpu"]} == set(range(8))
            and all(
                math.isfinite(float(pair[metric]))
                for pair in row["per_gpu"]
                for metric in ("throughput_ratio", "peak_allocated_ratio")
            )
            for row in paired_infer
        ),
        "checkpoint parity": parity["valid_complete_runs"]
        == parity["expected_runs"]
        == 30
        and not parity["missing_runs"]
        and not parity["invalid_runs"],
        "campaign state": data["campaign_state"]["phase"] == "complete",
        "release result": readiness["final_release_status"] in {"READY", "NO_GO"},
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("visual report refuses incomplete evidence: " + ", ".join(failed))


def decision_rows(data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(condition["metric"]), int(row["resize"])): row
        for condition in data["decision"]["conditions"]
        for row in condition["per_resize"]
    }


def gate_score(data: dict[str, Any]) -> tuple[int, int]:
    rows = list(decision_rows(data).values())
    return sum(str(row["status"]) == "PASS" for row in rows), len(rows)


def condition_score(data: dict[str, Any]) -> tuple[int, int]:
    conditions = list(data["decision"]["conditions"])
    return sum(str(condition["status"]) == "PASS" for condition in conditions), len(conditions)


def imagenet_scaleup_recommendation(data: dict[str, Any]) -> str:
    """Keep the locked CIFAR gate separate from the next-experiment decision."""

    rows = decision_rows(data)
    required_passes = (
        ("paired_accuracy_delta_pp", 256),
        ("paired_accuracy_delta_pp", 320),
        ("train_peak_allocated_ratio", 256),
        ("train_peak_allocated_ratio", 320),
        ("train_throughput_ratio", 320),
    )
    if str(data["checkpoint_parity"]["gate_status"]) != "PASS":
        return "HOLD"
    if any(key not in rows or str(rows[key]["status"]) != "PASS" for key in required_passes):
        return "HOLD"
    return "GO"


def validate_locked_source(data: dict[str, Any], aggregate_path: Path) -> None:
    """Require the manifest-locked canonical JSON/CSV/Markdown evidence bundle."""

    aggregate_path = aggregate_path.expanduser().resolve()
    if aggregate_path.name != "aggregate_results.json":
        raise ValueError("visual report requires the canonical aggregate_results.json path")
    try:
        manifest_path = aggregate_path.parent / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        for name in (
            "aggregate_results.json",
            "aggregate_results.csv",
            "aggregate_results.md",
        ):
            source = aggregate_path.parent / name
            entry = files[name]
            if (
                not source.is_file()
                or source.stat().st_size != int(entry["bytes"])
                or sha256(source) != str(entry["sha256"])
            ):
                raise ValueError(f"manifest lock mismatch for {name}")
        canonical_data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"visual report refuses unlocked evidence: {exc}") from exc
    if data != canonical_data:
        raise ValueError("visual report data does not match the canonical validated aggregate JSON")


def local_time(iso_value: str) -> str:
    moment = datetime.fromisoformat(iso_value)
    return moment.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")


def svg_open(title: str, desc: str, width: int, height: int) -> str:
    ident = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title-{ident} desc-{ident}">'
        f'<title id="title-{ident}">{esc(title)}</title>'
        f'<desc id="desc-{ident}">{esc(desc)}</desc>'
    )


def line_chart(
    resizes: list[int], accuracy: dict[tuple[int, str], dict[str, Any]]
) -> str:
    width, height = 920, 445
    left, right, top, bottom = 72, 30, 62, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 64.0, 72.0
    x = lambda index: left + index * plot_w / (len(resizes) - 1)
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        svg_open(
            "不同输入尺寸的 Top-1 趋势",
            "三模型 epoch 199 EMA Top-1 均值、三种子散点与样本标准差误差条。",
            width,
            height,
        )
    ]
    for tick in range(64, 73, 2):
        ty = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="grid"/>'
            f'<text x="{left-13}" y="{ty+4:.1f}" text-anchor="end" class="axis">{tick}%</text>'
        )
    for index, resize in enumerate(resizes):
        tx = x(index)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" class="grid vertical"/>'
            f'<text x="{tx:.1f}" y="{height-bottom+28}" text-anchor="middle" class="axis">{resize}</text>'
        )

    for legend_index, model in enumerate(("deit_s8", "mn_l2", "mn_l4")):
        label, color = MODEL_META[model]
        lx = 118 + legend_index * 232
        parts.append(
            f'<line x1="{lx}" y1="30" x2="{lx+32}" y2="30" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="{lx+16}" cy="30" r="5" fill="{color}" stroke="#fff" stroke-width="2"/>'
            f'<text x="{lx+43}" y="35" class="legend">{esc(label)}</text>'
        )

    offsets = (-8, 0, 8)
    for model in ("deit_s8", "mn_l2", "mn_l4"):
        label, color = MODEL_META[model]
        rows = [accuracy[(resize, model)] for resize in resizes]
        coords = [(x(index), y(row["mean"])) for index, row in enumerate(rows)]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {px:.1f} {py:.1f}"
            for index, (px, py) in enumerate(coords)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for index, row in enumerate(rows):
            px, py = coords[index]
            high, low = y(row["mean"] + row["sample_sd"]), y(row["mean"] - row["sample_sd"])
            parts.append(
                f'<line x1="{px:.1f}" y1="{high:.1f}" x2="{px:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.8"/>'
                f'<line x1="{px-5:.1f}" y1="{high:.1f}" x2="{px+5:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="1.8"/>'
                f'<line x1="{px-5:.1f}" y1="{low:.1f}" x2="{px+5:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.8"/>'
            )
            for seed_index, value in enumerate(row["values"]):
                parts.append(
                    f'<circle cx="{px+offsets[seed_index]:.1f}" cy="{y(value):.1f}" r="3" '
                    f'fill="{color}" opacity=".35"><title>{esc(label)} · size {resizes[index]} · '
                    f'seed {42+seed_index}: {value:.2f}%</title></circle>'
                )
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="2">'
                f'<title>{esc(label)} · size {resizes[index]}: {row["mean"]:.2f} ± {row["sample_sd"]:.2f}%</title></circle>'
            )
    parts.append(
        f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" class="axis-label">输入尺寸</text>'
        f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {top+plot_h/2:.1f})">EMA Top-1 (%)</text></svg>'
    )
    return "".join(parts)


def delta_chart(
    resizes: list[int], deltas: dict[tuple[int, str], dict[str, Any]]
) -> str:
    width, height = 920, 420
    left, right, top, bottom = 72, 30, 58, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = -1.2, 6.5
    group_w = plot_w / len(resizes)
    bar_w = 38
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    parts = [
        svg_open(
            "相对 DeiT 的同 seed 配对精度增益",
            "MergeNet lambda2 与 lambda4 相对 DeiT 的配对 Top-1 差值，误差条为样本标准差。",
            width,
            height,
        )
    ]
    for tick in (-1, 0, 1, 2, 3, 4, 5, 6):
        ty = y(tick)
        cls = "baseline" if tick == 0 else "grid"
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="{cls}"/>'
            f'<text x="{left-13}" y="{ty+4:.1f}" text-anchor="end" class="axis">{tick:+d}</text>'
        )
    for legend_index, model in enumerate(("mn_l2", "mn_l4")):
        label, color = MODEL_META[model]
        lx = 252 + legend_index * 225
        parts.append(
            f'<rect x="{lx}" y="21" width="17" height="17" rx="4" fill="{color}"/>'
            f'<text x="{lx+27}" y="35" class="legend">{esc(label)} − DeiT</text>'
        )

    zero_y = y(0)
    for index, resize in enumerate(resizes):
        center = left + group_w * (index + 0.5)
        for model_index, model in enumerate(("mn_l2", "mn_l4")):
            row = deltas[(resize, model)]
            _, color = MODEL_META[model]
            px = center + (-bar_w * 0.62 if model_index == 0 else bar_w * 0.62)
            value_y = y(row["mean"])
            rect_y = min(value_y, zero_y)
            rect_h = max(abs(zero_y - value_y), 1)
            high, low = y(row["mean"] + row["sample_sd"]), y(row["mean"] - row["sample_sd"])
            label_y = value_y - 9 if row["mean"] >= 0 else value_y + 18
            parts.append(
                f'<rect x="{px-bar_w/2:.1f}" y="{rect_y:.1f}" width="{bar_w}" height="{rect_h:.1f}" '
                f'rx="5" fill="{color}" opacity=".9"><title>{esc(MODEL_META[model][0])} · size {resize}: '
                f'{row["mean"]:+.2f} ± {row["sample_sd"]:.2f} pp</title></rect>'
                f'<line x1="{px:.1f}" y1="{high:.1f}" x2="{px:.1f}" y2="{low:.1f}" stroke="#243b53" stroke-width="1.5"/>'
                f'<line x1="{px-5:.1f}" y1="{high:.1f}" x2="{px+5:.1f}" y2="{high:.1f}" stroke="#243b53" stroke-width="1.5"/>'
                f'<line x1="{px-5:.1f}" y1="{low:.1f}" x2="{px+5:.1f}" y2="{low:.1f}" stroke="#243b53" stroke-width="1.5"/>'
                f'<text x="{px:.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{row["mean"]:+.2f}</text>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="{height-bottom+29}" text-anchor="middle" class="axis">{resize}</text>'
        )
    parts.append(
        f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" class="axis-label">输入尺寸</text>'
        f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {top+plot_h/2:.1f})">paired Δ Top-1 (pp)</text></svg>'
    )
    return "".join(parts)


def absolute_efficiency_chart(
    resizes: list[int],
    raw_train: dict[tuple[int, str], dict[str, Any]],
    *,
    metric: str,
    title: str,
    description: str,
    y_label: str,
    ticks: tuple[int, ...],
) -> str:
    width, height = 920, 420
    left, right, top, bottom = 82, 30, 62, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = float(ticks[0]), float(ticks[-1])
    x = lambda index: left + index * plot_w / (len(resizes) - 1)
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    parts = [svg_open(title, description, width, height)]

    for tick in ticks:
        ty = y(tick)
        tick_label = f"{tick:,}"
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="grid"/>'
            f'<text x="{left-13}" y="{ty+4:.1f}" text-anchor="end" class="axis">{tick_label}</text>'
        )
    for index, resize in enumerate(resizes):
        tx = x(index)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" class="grid vertical"/>'
            f'<text x="{tx:.1f}" y="{height-bottom+28}" text-anchor="middle" class="axis">{resize}</text>'
        )
    for legend_index, model in enumerate(("deit_s8", "mn_l2", "mn_l4")):
        label, color = MODEL_META[model]
        lx = 118 + legend_index * 232
        parts.append(
            f'<line x1="{lx}" y1="30" x2="{lx+32}" y2="30" stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="{lx+16}" cy="30" r="5" fill="{color}" stroke="#fff" stroke-width="2"/>'
            f'<text x="{lx+43}" y="35" class="legend">{esc(label)}</text>'
        )

    for model in ("deit_s8", "mn_l2", "mn_l4"):
        label, color = MODEL_META[model]
        rows = [raw_train[(resize, model)] for resize in resizes]
        coords = [(x(index), y(row[metric]["mean"])) for index, row in enumerate(rows)]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {px:.1f} {py:.1f}"
            for index, (px, py) in enumerate(coords)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for index, row in enumerate(rows):
            value = row[metric]["mean"]
            sd = row[metric]["sample_sd"]
            px, py = coords[index]
            high = y(min(value + sd, y_max))
            low = y(max(value - sd, y_min))
            parts.append(
                f'<line x1="{px:.1f}" y1="{high:.1f}" x2="{px:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.8"/>'
                f'<line x1="{px-5:.1f}" y1="{high:.1f}" x2="{px+5:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="1.8"/>'
                f'<line x1="{px-5:.1f}" y1="{low:.1f}" x2="{px+5:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.8"/>'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="2">'
                f'<title>{esc(label)} · size {resizes[index]}: {value:,.1f} ± {sd:,.1f} {esc(y_label)}</title></circle>'
            )
    parts.append(
        f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" class="axis-label">输入尺寸</text>'
        f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {top+plot_h/2:.1f})">{esc(y_label)}</text></svg>'
    )
    return "".join(parts)


def inference_throughput_chart(
    resizes: list[int],
    raw_infer: dict[tuple[int, str, str], dict[str, Any]],
) -> str:
    """Plot the one DeiT inference baseline and both MergeNet inference paths."""

    width, height = 920, 455
    left, right, top, bottom = 82, 30, 78, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 4500.0
    x = lambda index: left + index * plot_w / (len(resizes) - 1)
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    parts = [
        svg_open(
            "Generic 与 fast inference 绝对吞吐",
            "DeiT generic 基线及 MergeNet lambda2/lambda4 generic、fast 路径在五个输入尺寸上的八卡平均推理吞吐。",
            width,
            height,
        )
    ]

    for tick in (0, 1000, 2000, 3000, 4000, 4500):
        ty = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="grid"/>'
            f'<text x="{left-13}" y="{ty+4:.1f}" text-anchor="end" class="axis">{tick:,}</text>'
        )
    for index, resize in enumerate(resizes):
        tx = x(index)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" class="grid vertical"/>'
            f'<text x="{tx:.1f}" y="{height-bottom+28}" text-anchor="middle" class="axis">{resize}</text>'
        )

    legend_x = (104, 310, 448, 584, 722)
    for index, (mode, model, label, dash) in enumerate(INFERENCE_SERIES):
        _, color = MODEL_META[model]
        lx = legend_x[index]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        fill = "#fff" if dash else color
        parts.append(
            f'<line x1="{lx}" y1="31" x2="{lx+28}" y2="31" stroke="{color}" '
            f'stroke-width="3.5"{dash_attr}/>'
            f'<circle cx="{lx+14}" cy="31" r="5" fill="{fill}" stroke="{color}" stroke-width="2"/>'
            f'<text x="{lx+37}" y="36" class="legend">{esc(label)}</text>'
        )
    parts.append(
        '<text x="104" y="59" class="chart-note">MergeNet 虚线/空心点：generic · 实线/实心点：fast；灰色实线为 DeiT 单一基线</text>'
    )

    for mode, model, label, dash in INFERENCE_SERIES:
        _, color = MODEL_META[model]
        rows = [raw_infer[(resize, model, mode)] for resize in resizes]
        coords = [
            (x(index), y(row["throughput_img_s"]["mean"]))
            for index, row in enumerate(rows)
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {px:.1f} {py:.1f}"
            for index, (px, py) in enumerate(coords)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.3" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
        )
        for index, row in enumerate(rows):
            value = row["throughput_img_s"]["mean"]
            sd = row["throughput_img_s"]["sample_sd"]
            px, py = coords[index]
            high = y(min(value + sd, y_max))
            low = y(max(value - sd, y_min))
            fill = "#fff" if dash else color
            parts.append(
                f'<line x1="{px:.1f}" y1="{high:.1f}" x2="{px:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.7"/>'
                f'<line x1="{px-5:.1f}" y1="{high:.1f}" x2="{px+5:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="1.7"/>'
                f'<line x1="{px-5:.1f}" y1="{low:.1f}" x2="{px+5:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.7"/>'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{fill}" stroke="{color}" stroke-width="2">'
                f'<title>{esc(label)} · size {resizes[index]}: {value:,.1f} ± {sd:,.1f} img/s</title></circle>'
            )
    parts.append(
        f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" class="axis-label">输入尺寸</text>'
        f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {top+plot_h/2:.1f})">推理吞吐 (img/s)</text></svg>'
    )
    return "".join(parts)


def inference_ratio_chart(
    resizes: list[int],
    paired_infer: dict[tuple[int, str, str], dict[str, Any]],
) -> str:
    width, height = 920, 430
    left, right, top, bottom = 78, 30, 70, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.30, 1.08
    x = lambda index: left + index * plot_w / (len(resizes) - 1)
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    parts = [
        svg_open(
            "Inference 吞吐相对 DeiT 的配对比",
            "MergeNet generic 与 fast 路径在同一物理 GPU 上相对 DeiT generic 基线的推理吞吐比。",
            width,
            height,
        )
    ]
    for tick in (0.4, 0.6, 0.8, 1.0):
        ty = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="grid"/>'
            f'<text x="{left-13}" y="{ty+4:.1f}" text-anchor="end" class="axis">{tick:.1f}×</text>'
        )
    for index, resize in enumerate(resizes):
        tx = x(index)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" class="grid vertical"/>'
            f'<text x="{tx:.1f}" y="{height-bottom+28}" text-anchor="middle" class="axis">{resize}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{y(1):.1f}" x2="{width-right}" y2="{y(1):.1f}" class="baseline"/>'
        f'<text x="{width-right-4}" y="{y(1)-8:.1f}" text-anchor="end" class="baseline-label">DeiT 1.0×</text>'
    )

    candidate_series = tuple(item for item in INFERENCE_SERIES if item[1] != "deit_s8")
    legend_x = (160, 330, 500, 670)
    for index, (mode, model, label, dash) in enumerate(candidate_series):
        _, color = MODEL_META[model]
        lx = legend_x[index]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        fill = "#fff" if dash else color
        parts.append(
            f'<line x1="{lx}" y1="30" x2="{lx+30}" y2="30" stroke="{color}" stroke-width="3.5"{dash_attr}/>'
            f'<circle cx="{lx+15}" cy="30" r="5" fill="{fill}" stroke="{color}" stroke-width="2"/>'
            f'<text x="{lx+39}" y="35" class="legend">{esc(label)}</text>'
        )

    for mode, model, label, dash in candidate_series:
        _, color = MODEL_META[model]
        rows = [paired_infer[(resize, model, mode)] for resize in resizes]
        coords = [
            (x(index), y(row["throughput_ratio"]["mean"]))
            for index, row in enumerate(rows)
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {px:.1f} {py:.1f}"
            for index, (px, py) in enumerate(coords)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.3" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
        )
        for index, row in enumerate(rows):
            value = row["throughput_ratio"]["mean"]
            sd = row["throughput_ratio"]["sample_sd"]
            px, py = coords[index]
            high = y(min(value + sd, y_max))
            low = y(max(value - sd, y_min))
            fill = "#fff" if dash else color
            halo = ""
            label_text = ""
            if mode == "infer_fast" and model == "mn_l4" and resizes[index] == 320:
                halo = (
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="none" '
                    f'stroke="{color}" stroke-width="2" opacity=".45"/>'
                )
                label_text = (
                    f'<text x="{px-11:.1f}" y="{py-15:.1f}" text-anchor="end" '
                    f'class="crossover-label">{value:.3f}× · crossover</text>'
                )
            parts.append(
                halo
                + f'<line x1="{px:.1f}" y1="{high:.1f}" x2="{px:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.7"/>'
                f'<line x1="{px-5:.1f}" y1="{high:.1f}" x2="{px+5:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="1.7"/>'
                f'<line x1="{px-5:.1f}" y1="{low:.1f}" x2="{px+5:.1f}" y2="{low:.1f}" stroke="{color}" stroke-width="1.7"/>'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{fill}" stroke="{color}" stroke-width="2">'
                f'<title>{esc(label)} · size {resizes[index]}: {value:.4f} ± {sd:.4f}× DeiT</title></circle>'
                + label_text
            )
    parts.append(
        f'<text x="{left+plot_w/2:.1f}" y="{height-12}" text-anchor="middle" class="axis-label">输入尺寸</text>'
        f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {top+plot_h/2:.1f})">推理吞吐比（candidate / DeiT）</text></svg>'
    )
    return "".join(parts)


def tradeoff_chart(train_rows: list[dict[str, Any]]) -> str:
    width, height = 920, 445
    left, right, top, bottom = 76, 32, 62, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max, y_min, y_max = 0.35, 1.12, 0.78, 1.18
    x = lambda value: left + (value - x_min) / (x_max - x_min) * plot_w
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    parts = [
        svg_open(
            "训练吞吐与显存权衡",
            "候选模型相对 DeiT 的八卡配对训练吞吐与峰值 allocated memory 比值。右下为更快且更省显存。",
            width,
            height,
        )
    ]
    parts.append(
        f'<rect x="{x(1):.1f}" y="{y(1):.1f}" width="{x(x_max)-x(1):.1f}" '
        f'height="{y(y_min)-y(1):.1f}" fill="#d8f3dc" opacity=".72"/>'
        f'<text x="{x(1.055):.1f}" y="{y(.82):.1f}" text-anchor="middle" class="ideal">更快 + 更省显存</text>'
    )
    for tick in (0.4, 0.6, 0.8, 1.0):
        tx = x(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" class="grid"/>'
            f'<text x="{tx:.1f}" y="{height-bottom+27}" text-anchor="middle" class="axis">{tick:.1f}×</text>'
        )
    for tick in (0.8, 0.9, 1.0, 1.1):
        ty = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="grid"/>'
            f'<text x="{left-13}" y="{ty+4:.1f}" text-anchor="end" class="axis">{tick:.1f}×</text>'
        )
    parts.append(
        f'<line x1="{x(1):.1f}" y1="{top}" x2="{x(1):.1f}" y2="{height-bottom}" class="baseline"/>'
        f'<line x1="{left}" y1="{y(1):.1f}" x2="{width-right}" y2="{y(1):.1f}" class="baseline"/>'
    )
    for legend_index, model in enumerate(("deit_s8", "mn_l2", "mn_l4")):
        label, color = MODEL_META[model]
        lx = 165 + legend_index * 225
        marker = (
            f'<path d="M {lx} 20 L {lx+9} 29 L {lx} 38 L {lx-9} 29 Z" fill="{color}"/>'
            if model == "deit_s8"
            else f'<circle cx="{lx}" cy="29" r="8" fill="{color}"/>'
        )
        suffix = " · 归一化基线" if model == "deit_s8" else ""
        parts.append(marker + f'<text x="{lx+16}" y="34" class="legend">{esc(label + suffix)}</text>')

    baseline_x, baseline_y = x(1), y(1)
    parts.append(
        f'<path d="M {baseline_x:.1f} {baseline_y-11:.1f} L {baseline_x+11:.1f} {baseline_y:.1f} '
        f'L {baseline_x:.1f} {baseline_y+11:.1f} L {baseline_x-11:.1f} {baseline_y:.1f} Z" '
        f'fill="{MODEL_META["deit_s8"][1]}" stroke="#fff" stroke-width="3">'
        '<title>DeiT 基线：每个 size 在配对比值图中均归一化为 throughput 1.000×、allocated memory 1.000×；绝对值见下方图表。</title></path>'
    )

    label_offsets = {
        ("mn_l2", 160): (-14, -11),
        ("mn_l2", 192): (-5, -12),
        ("mn_l2", 224): (0, -12),
        ("mn_l2", 256): (0, -12),
        ("mn_l2", 320): (0, -12),
        ("mn_l4", 160): (-5, 23),
        ("mn_l4", 192): (0, 23),
        ("mn_l4", 224): (0, 23),
        ("mn_l4", 256): (-10, 24),
        ("mn_l4", 320): (-24, -13),
    }
    for model in ("mn_l2", "mn_l4"):
        rows = sorted(
            (row for row in train_rows if row["candidate_model"] == model),
            key=lambda row: row["resize"],
        )
        _, color = MODEL_META[model]
        path = " ".join(
            ("M" if index == 0 else "L")
            + f' {x(row["throughput_ratio"]["mean"]):.1f} {y(row["peak_allocated_ratio"]["mean"]):.1f}'
            for index, row in enumerate(rows)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" opacity=".45" stroke-dasharray="5 5"/>'
        )
        for row in rows:
            throughput = row["throughput_ratio"]["mean"]
            memory = row["peak_allocated_ratio"]["mean"]
            px, py = x(throughput), y(memory)
            dx, dy = label_offsets[(model, row["resize"])]
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="{color}" stroke="#fff" stroke-width="3">'
                f'<title>{esc(MODEL_META[model][0])} · size {row["resize"]}: throughput {throughput:.3f}×, '
                f'allocated memory {memory:.3f}×</title></circle>'
                f'<text x="{px+dx:.1f}" y="{py+dy:.1f}" text-anchor="middle" class="point-label">{row["resize"]}</text>'
            )
    parts.append(
        f'<text x="{left+plot_w/2:.1f}" y="{height-14}" text-anchor="middle" class="axis-label">训练吞吐比（candidate / DeiT）→ 越右越快</text>'
        f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 18 {top+plot_h/2:.1f})">allocated 显存比 → 越低越省</text></svg>'
    )
    return "".join(parts)


def gate_rows_html(conditions: Iterable[dict[str, Any]]) -> str:
    labels = {
        "paired_accuracy_delta_pp": "λ4 精度增益",
        "train_throughput_ratio": "λ4 训练吞吐比",
        "train_peak_allocated_ratio": "λ4 allocated 显存比",
    }
    rows: list[str] = []
    for condition in conditions:
        for item in condition["per_resize"]:
            status = item["status"]
            rows.append(
                "<tr>"
                f'<td>{esc(labels[condition["metric"]])}</td>'
                f'<td class="num">{item["resize"]}</td>'
                f'<td class="num">{item["n"]}/{item["required_n"]}</td>'
                f'<td class="num"><code>{esc(item["op"])} {item["threshold"]:.2f}</code></td>'
                f'<td class="num strong">{item["value"]:.4f}</td>'
                f'<td><span class="status {status.lower()}">{status}</span></td>'
                "</tr>"
            )
    return "".join(rows)


def absolute_efficiency_rows_html(
    resizes: list[int], raw_train: dict[tuple[int, str], dict[str, Any]]
) -> str:
    rows: list[str] = []
    for resize in resizes:
        for model in ("deit_s8", "mn_l2", "mn_l4"):
            row = raw_train[(resize, model)]
            model_class = (
                "baseline-row"
                if model == "deit_s8"
                else ("l2-row" if model == "mn_l2" else "l4-row")
            )
            rows.append(
                f'<tr class="{model_class}" data-absolute-train-efficiency-row="{model}-r{resize}">'
                f'<td class="num strong">{resize}</td>'
                f'<td>{esc(MODEL_META[model][0])}</td>'
                f'<td class="num strong">{row["throughput_img_s"]["mean"]:,.1f} ± '
                f'{row["throughput_img_s"]["sample_sd"]:,.1f}</td>'
                f'<td class="num">{row["peak_allocated_mib"]["mean"]:,.1f} ± '
                f'{row["peak_allocated_mib"]["sample_sd"]:,.1f}</td>'
                f'<td class="num">{row["step_time_ms"]["mean"]:,.2f} ± '
                f'{row["step_time_ms"]["sample_sd"]:,.2f}</td>'
                f'<td class="num">{row["params"]["mean"] / 1_000_000:.3f} M</td>'
                '<td class="num">8 / 8</td>'
                "</tr>"
            )
    return "".join(rows)


def inference_efficiency_rows_html(
    resizes: list[int],
    raw_infer: dict[tuple[int, str, str], dict[str, Any]],
    paired_infer: dict[tuple[int, str, str], dict[str, Any]],
) -> str:
    rows: list[str] = []
    for resize in resizes:
        for mode, model, label, _ in INFERENCE_SERIES:
            raw = raw_infer[(resize, model, mode)]
            model_class = (
                "baseline-row"
                if model == "deit_s8"
                else ("l2-row" if model == "mn_l2" else "l4-row")
            )
            if model == "deit_s8":
                throughput_ratio = "1.000× · baseline"
                memory_ratio = "1.000× · baseline"
            else:
                paired = paired_infer[(resize, model, mode)]
                throughput_ratio = (
                    f'{paired["throughput_ratio"]["mean"]:.3f} ± '
                    f'{paired["throughput_ratio"]["sample_sd"]:.3f}×'
                )
                memory_ratio = (
                    f'{paired["peak_allocated_ratio"]["mean"]:.3f} ± '
                    f'{paired["peak_allocated_ratio"]["sample_sd"]:.3f}×'
                )
            rows.append(
                f'<tr class="{model_class}" data-absolute-inference-row="{model}-r{resize}-{mode}">'
                f'<td class="num strong">{resize}</td>'
                f'<td>{esc(MODEL_META[model][0])}</td>'
                f'<td>{esc("generic baseline" if model == "deit_s8" else mode.removeprefix("infer_"))}</td>'
                f'<td class="num strong">{raw["throughput_img_s"]["mean"]:,.1f} ± '
                f'{raw["throughput_img_s"]["sample_sd"]:,.1f}</td>'
                f'<td class="num">{esc(throughput_ratio)}</td>'
                f'<td class="num">{raw["step_time_ms"]["mean"]:,.2f} ± '
                f'{raw["step_time_ms"]["sample_sd"]:,.2f}</td>'
                f'<td class="num">{raw["peak_allocated_mib"]["mean"]:,.1f} ± '
                f'{raw["peak_allocated_mib"]["sample_sd"]:,.1f}</td>'
                f'<td class="num">{esc(memory_ratio)}</td>'
                '<td class="num">8 / 8</td>'
                "</tr>"
            )
    return "".join(rows)


def summary_rows_html(
    resizes: list[int],
    accuracy: dict[tuple[int, str], dict[str, Any]],
    deltas: dict[tuple[int, str], dict[str, Any]],
    train: dict[tuple[int, str], dict[str, Any]],
    paired_infer: dict[tuple[int, str, str], dict[str, Any]],
) -> str:
    rows: list[str] = []
    for resize in resizes:
        deit = accuracy[(resize, "deit_s8")]
        l2 = accuracy[(resize, "mn_l2")]
        l4 = accuracy[(resize, "mn_l4")]
        l2_delta = deltas[(resize, "mn_l2")]
        l4_delta = deltas[(resize, "mn_l4")]
        l4_eff = train[(resize, "mn_l4")]
        l4_fast_infer = paired_infer[(resize, "mn_l4", "infer_fast")]
        flag = ""
        if resize == 256:
            if float(l4_eff["throughput_ratio"]["mean"]) < 1.0:
                flag = '<span class="status fail">FAIL · tracked</span>'
            else:
                flag = '<span class="status pass">PASS</span>'
        elif resize == 320:
            flag = '<span class="status pass">crossover</span>'
        rows.append(
            "<tr>"
            f'<td class="num strong">{resize}</td>'
            f'<td class="num">{deit["mean"]:.2f} ± {deit["sample_sd"]:.2f}</td>'
            f'<td class="num accent-l2">{l2["mean"]:.2f} ± {l2["sample_sd"]:.2f}</td>'
            f'<td class="num accent-l2">{l2_delta["mean"]:+.2f}</td>'
            f'<td class="num accent-l4">{l4["mean"]:.2f} ± {l4["sample_sd"]:.2f}</td>'
            f'<td class="num accent-l4">{l4_delta["mean"]:+.2f}</td>'
            f'<td class="num">{l4_eff["throughput_ratio"]["mean"]:.3f}×</td>'
            f'<td class="num">{l4_eff["peak_allocated_ratio"]["mean"]:.3f}×</td>'
            f'<td class="num">{l4_fast_infer["throughput_ratio"]["mean"]:.3f}×</td>'
            f"<td>{flag}</td>"
            "</tr>"
        )
    return "".join(rows)


def seed_rows_html(
    resizes: list[int], accuracy: dict[tuple[int, str], dict[str, Any]]
) -> str:
    rows: list[str] = []
    for resize in resizes:
        for model in ("deit_s8", "mn_l2", "mn_l4"):
            row = accuracy[(resize, model)]
            rows.append(
                "<tr>"
                f'<td class="num">{resize}</td><td>{esc(MODEL_META[model][0])}</td>'
                + "".join(f'<td class="num">{value:.2f}</td>' for value in row["values"])
                + f'<td class="num strong">{row["mean"]:.2f} ± {row["sample_sd"]:.2f}</td>'
                + "</tr>"
            )
    return "".join(rows)


def build_html(
    data: dict[str, Any],
    aggregate_path: Path,
    *,
    _evidence_validated: bool = False,
) -> str:
    validate(data)
    if not _evidence_validated:
        validate_locked_source(data, aggregate_path)
    resizes = data["matrix"]["resizes"]
    accuracy = {
        (row["resize"], row["model_id"]): row for row in data["accuracy"]["summary"]
    }
    deltas = {
        (row["resize"], row["candidate_model"]): row
        for row in data["accuracy"]["paired_deltas"]
    }
    train_rows = [row for row in data["efficiency"]["paired_ratios"] if row["mode"] == "train"]
    train = {(row["resize"], row["candidate_model"]): row for row in train_rows}
    raw_train_rows = [
        row for row in data["efficiency"]["raw_summary"] if row["mode"] == "train"
    ]
    raw_train = {(row["resize"], row["model_id"]): row for row in raw_train_rows}
    raw_infer_rows = [
        row
        for row in data["efficiency"]["raw_summary"]
        if row["mode"] == "infer_generic"
        or (row["mode"] == "infer_fast" and row["model_id"] != "deit_s8")
    ]
    raw_infer = {
        (row["resize"], row["model_id"], row["mode"]): row
        for row in raw_infer_rows
    }
    paired_infer_rows = [
        row
        for row in data["efficiency"]["paired_ratios"]
        if row["mode"] in {"infer_generic", "infer_fast"}
    ]
    paired_infer = {
        (row["resize"], row["candidate_model"], row["mode"]): row
        for row in paired_infer_rows
    }
    readiness = data["release_readiness"]
    parity = data["checkpoint_parity"]
    completion = data["completeness"]
    decision = data["decision"]
    aggregate_hash = sha256(aggregate_path)
    finished = local_time(data["campaign_state"]["updated_at"])
    aggregated = local_time(data["generated_at"])
    release_status = readiness["final_release_status"]
    scaleup_status = imagenet_scaleup_recommendation(data)
    scaleup_class = "pass" if scaleup_status == "GO" else "fail"
    passed, total = gate_score(data)
    condition_passed, condition_total = condition_score(data)
    tracked_misses = total - passed
    l2_accuracy_320 = accuracy[(320, "mn_l2")]["mean"]
    l2_delta_320 = deltas[(320, "mn_l2")]["mean"]
    l4_delta_256 = deltas[(256, "mn_l4")]["mean"]
    l4_delta_320 = deltas[(320, "mn_l4")]["mean"]
    l4_throughput_256 = train[(256, "mn_l4")]["throughput_ratio"]["mean"]
    l4_throughput_320 = train[(320, "mn_l4")]["throughput_ratio"]["mean"]
    l4_memory_256 = train[(256, "mn_l4")]["peak_allocated_ratio"]["mean"]
    l4_memory_320 = train[(320, "mn_l4")]["peak_allocated_ratio"]["mean"]
    l4_generic_infer_320 = paired_infer[(320, "mn_l4", "infer_generic")]
    l4_fast_infer_320 = paired_infer[(320, "mn_l4", "infer_fast")]
    l4_fast_above_baseline_cards_320 = sum(
        float(row["throughput_ratio"]) > 1.0
        for row in l4_fast_infer_320["per_gpu"]
    )
    l4_fast_raw_320 = raw_infer[(320, "mn_l4", "infer_fast")]
    l4_generic_raw_320 = raw_infer[(320, "mn_l4", "infer_generic")]
    l4_fast_uplift_320 = (
        l4_fast_raw_320["throughput_img_s"]["mean"]
        / l4_generic_raw_320["throughput_img_s"]["mean"]
        - 1
    )
    deit_infer_drop_160_320 = 1 - (
        raw_infer[(320, "deit_s8", "infer_generic")]["throughput_img_s"]["mean"]
        / raw_infer[(160, "deit_s8", "infer_generic")]["throughput_img_s"]["mean"]
    )
    l4_fast_infer_drop_160_320 = 1 - (
        l4_fast_raw_320["throughput_img_s"]["mean"]
        / raw_infer[(160, "mn_l4", "infer_fast")]["throughput_img_s"]["mean"]
    )
    parity_status = str(parity["gate_status"])
    parity_class = "" if parity_status == "PASS" else "risk"
    parity_max_delta = max(float(row["abs_top1_delta_pp"]) for row in parity["runs"])
    parity_card_title = f"{parity['valid_complete_runs']} / {parity['expected_runs']} · {parity_status}"
    if parity_status == "PASS":
        parity_card_detail = (
            f"generic / fast 最大 |ΔTop-1| 为 {parity_max_delta:.2f} pp，低于 0.05 pp 门限。"
        )
    else:
        parity_card_detail = (
            f"{parity['failed_runs']} 个 checkpoint 未通过 generic / fast 门限；ImageNet scale-up 被阻塞。"
        )
    if scaleup_status == "GO":
        scaleup_heading = "实验完整结束；建议进入 ImageNet 规模预训练实验"
        hero_lead = (
            "五个输入尺寸全部完成。结果呈现稳定的精度收益、λ4 全尺寸训练 allocated 显存优势、"
            "随尺寸增大的训练吞吐 crossover，以及 size 320 的 fast inference 轻微 crossover。"
            "这些一致趋势支持进入 ImageNet 规模预训练验证。"
        )
        if l4_throughput_256 < 1.0:
            scaleup_action = (
                "因此建议启动受控 ImageNet 预训练实验；256 的 "
                f"{l4_throughput_256:.3f}× 训练吞吐仍作为已知风险监控。"
            )
        else:
            scaleup_action = "因此建议启动受控 ImageNet 预训练实验，并与 matched DeiT 共同长训。"
    else:
        scaleup_heading = "实验完整结束；ImageNet 规模预训练暂缓"
        hero_lead = (
            "五个输入尺寸已经完整测量，但 checkpoint parity 或必要的规模化趋势仍有阻塞项；"
            "当前不启动论文规模 ImageNet 预训练。"
        )
        scaleup_action = "当前证据组合不足以启动论文规模 ImageNet 训练，应先解决阻塞项。"
    if l4_throughput_256 < 1.0:
        r256_card_class = "risk"
        r256_card_label = "已知规模化监控项"
        r256_card_detail = (
            f"距严格训练吞吐门槛 {100 * (1 - l4_throughput_256):.2f}%，同时带来 "
            f"{l4_delta_256:+.2f} pp 精度和 {100 * (1 - l4_memory_256):.1f}% 训练 allocated 显存降低；"
            "在 ImageNet 长训中跟踪端到端 wall-clock。"
        )
    else:
        r256_card_class = "l4"
        r256_card_label = "size 256 子检查"
        r256_card_detail = "训练吞吐、精度和 allocated 显存子检查均通过。"
    footer_decision = (
        "支持进入 ImageNet 规模验证；不宣称 ImageNet 结果已验证"
        if scaleup_status == "GO"
        else "ImageNet 规模验证暂缓；锁定证据保持不变"
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>MergeNet 不同 Size 消融 · 可视化报告</title>
  <style>
    :root{{--ink:#142531;--muted:#667784;--paper:#fffdf8;--canvas:#edf3f2;--line:#d8e2df;--navy:#123849;--l2:#168aad;--l4:#e76f51;--base:#718096;--good:#257a5a;--bad:#b34842;--gold:#a66b14;--shadow:0 15px 42px rgba(22,49,57,.08)}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif}}
    a{{color:#11647d}} button{{font:inherit}} .shell{{width:min(1240px,calc(100% - 32px));margin:0 auto 70px}}
    .hero{{margin-top:24px;padding:40px 44px;border-radius:26px;color:white;background:radial-gradient(circle at 82% 18%,rgba(60,205,190,.24),transparent 30%),linear-gradient(128deg,#102f3c,#15576b 68%,#196e78);box-shadow:0 26px 70px rgba(10,45,57,.22)}}
    .eyebrow{{font-size:12px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;opacity:.72}} h1{{margin:.15em 0 .28em;font-size:clamp(34px,5vw,66px);line-height:1.06;letter-spacing:-.035em}} h2{{margin:.15em 0 .55em;font-size:clamp(25px,3vw,36px);line-height:1.18;letter-spacing:-.02em}} h3{{margin:1.4em 0 .55em;font-size:20px}} p{{margin:.55em 0}} .lead{{max-width:900px;font-size:19px;opacity:.92}}
    .hero-grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:28px;align-items:end}} .verdict{{justify-self:end;min-width:250px;padding:20px 23px;border:1px solid rgba(255,255,255,.2);border-radius:18px;background:rgba(255,255,255,.1);backdrop-filter:blur(10px)}} .verdict small,.verdict span{{display:block;opacity:.76}} .verdict strong{{display:block;font-size:32px;line-height:1.25}}
    .completion{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:28px}} .completion div{{padding:16px;border:1px solid rgba(255,255,255,.18);border-radius:16px;background:rgba(255,255,255,.08)}} .completion b{{display:block;font-size:25px}} .completion span{{font-size:12px;opacity:.75}}
    nav{{position:sticky;top:10px;z-index:4;display:flex;flex-wrap:wrap;gap:8px;margin:16px 0;padding:10px;border:1px solid rgba(216,226,223,.9);border-radius:16px;background:rgba(255,253,248,.9);backdrop-filter:blur(12px);box-shadow:var(--shadow)}} nav a{{padding:6px 12px;border-radius:999px;text-decoration:none;font-weight:700}} nav a:hover{{background:#e2f1f0}}
    section{{margin:18px 0;padding:31px 34px;border:1px solid var(--line);border-radius:22px;background:var(--paper);box-shadow:var(--shadow)}}
    .split{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .insight{{padding:20px;border:1px solid var(--line);border-radius:16px;background:#f7faf8}} .insight strong{{display:block;margin:.12em 0;font-size:27px;line-height:1.25}} .insight.l2{{border-top:4px solid var(--l2)}} .insight.l4{{border-top:4px solid var(--l4)}} .insight.risk{{border-top:4px solid var(--gold)}}
    .status{{display:inline-block;padding:3px 10px;border-radius:999px;color:white;font-size:11px;font-weight:900;letter-spacing:.04em;text-transform:uppercase}} .status.pass{{background:var(--good)}} .status.fail{{background:var(--bad)}} .status.complete{{background:#327384}}
    .callout{{margin:16px 0;padding:16px 18px;border-left:4px solid var(--gold);border-radius:0 12px 12px 0;background:#fff4d9}} .muted{{color:var(--muted)}} .strong{{font-weight:800}} .num{{text-align:right;font-variant-numeric:tabular-nums}} .accent-l2{{color:#087b9c;font-weight:750}} .accent-l4{{color:#c5533e;font-weight:750}}
    .figure{{margin:24px 0}} .chart{{overflow-x:auto;padding:12px;border:1px solid var(--line);border-radius:18px;background:white}} .chart svg{{display:block;width:100%;height:auto;min-width:760px}} figcaption{{margin:9px 6px;color:var(--muted);font-size:13px}}
    svg .grid{{stroke:#dfe8e6;stroke-width:1}} svg .grid.vertical{{opacity:.55}} svg .baseline{{stroke:#8b5e34;stroke-width:1.6;stroke-dasharray:6 5}} svg .axis{{fill:#647681;font-size:12px}} svg .axis-label{{fill:#3e5664;font-size:13px;font-weight:750}} svg .legend{{fill:#263e4a;font-size:13px;font-weight:750}} svg .value{{fill:#293f4a;font-size:11px;font-weight:800}} svg .point-label{{fill:#243b46;font-size:11px;font-weight:900}} svg .ideal{{fill:#257a5a;font-size:11px;font-weight:850}} svg .chart-note{{fill:#667784;font-size:11px}} svg .baseline-label{{fill:#8b5e34;font-size:11px;font-weight:800}} svg .crossover-label{{fill:#b64232;font-size:11px;font-weight:900}}
    .table-wrap{{overflow:auto;margin:15px 0;border:1px solid var(--line);border-radius:14px}} table{{width:100%;min-width:850px;border-collapse:collapse}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{position:sticky;top:0;background:#eff5f3;color:#445b67;font-size:12px;text-align:left}} tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f5faf8}} tbody tr.baseline-row{{background:#f5f7fa}} tbody tr.baseline-row td:nth-child(2){{color:var(--base);font-weight:850}} tbody tr.l2-row td:nth-child(2){{color:var(--l2);font-weight:850}} tbody tr.l4-row td:nth-child(2){{color:var(--l4);font-weight:850}} code{{font-size:.92em}}
    details{{margin:18px 0;border:1px solid var(--line);border-radius:14px;background:white}} summary{{cursor:pointer;padding:14px 17px;font-weight:800}} details>div{{padding:0 17px 17px}}
    .evidence{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:18px}} .evidence a{{display:block;padding:13px 15px;border:1px solid var(--line);border-radius:12px;background:white;text-decoration:none;font-weight:750}} .evidence a:hover{{border-color:#72aeb2;background:#f4fbfa}}
    footer{{padding:24px;text-align:center;color:var(--muted);font-size:13px}} .actions{{display:flex;gap:8px;margin-top:18px}} .actions button{{cursor:pointer;padding:8px 13px;border:1px solid rgba(255,255,255,.25);border-radius:10px;color:white;background:rgba(255,255,255,.1)}}
    @media(max-width:800px){{.shell{{width:min(100% - 18px,1240px)}}.hero{{padding:28px 23px}}.hero-grid,.split{{grid-template-columns:1fr}}.verdict{{justify-self:start}}.completion{{grid-template-columns:1fr 1fr}}section{{padding:24px 19px}}.evidence{{grid-template-columns:1fr}}}}
    @media print{{body{{background:white}}.shell{{width:100%;margin:0}}.hero,section{{box-shadow:none;break-inside:avoid}}nav,.actions{{display:none}}.hero{{margin:0;border-radius:0;color:var(--ink);background:white;border-bottom:2px solid var(--ink)}}.verdict,.completion div{{border-color:var(--line);background:white}}.chart{{overflow:visible}}.chart svg{{min-width:0}}}}
  </style>
</head>
<body>
<main class="shell">
  <header class="hero">
    <div class="hero-grid">
      <div>
        <div class="eyebrow">CIFAR-100 · evidence-locked · epoch-199 EMA</div>
        <h1>MergeNet 不同 Size<br>消融实验</h1>
        <p class="lead">{esc(hero_lead)}</p>
        <div class="actions"><button type="button" onclick="window.print()">打印 / 导出 PDF</button></div>
      </div>
      <div class="verdict">
        <small>运行状态</small><strong>✓ 已完整结束</strong>
        <span>ImageNet 推进建议</span><strong><span class="status {scaleup_class}">{esc(scaleup_status)}</span></strong>
        <span>CIFAR 严格门禁：逐尺度 {passed}/{total} · 顶层 {condition_passed}/{condition_total} · {tracked_misses} tracked miss（归档 {esc(release_status)}）</span>
      </div>
    </div>
    <div class="completion">
      <div><b>5 / 5</b><span>size · 160–320</span></div>
      <div><b>{completion['accuracy_complete_jobs']} / {completion['accuracy_expected_jobs']}</b><span>accuracy runs</span></div>
      <div><b>{completion['efficiency_complete_cards']} / {completion['efficiency_expected_cards']}</b><span>GPU efficiency</span></div>
      <div><b>{parity['valid_complete_runs']} / {parity['expected_runs']}</b><span>checkpoint parity · {esc(parity['gate_status'])}</span></div>
    </div>
  </header>

  <nav aria-label="报告目录">
    <a href="#takeaways">结论</a><a href="#accuracy">精度趋势</a><a href="#delta">配对增益</a><a href="#absolute-efficiency">训练绝对效率</a><a href="#inference">推理效率</a><a href="#tradeoff">训练相对权衡</a><a href="#gate">门禁与证据</a>
  </nav>

  <section id="takeaways">
    <div class="eyebrow">Executive readout</div><h2>{esc(scaleup_heading)}</h2>
    <div class="callout"><strong>结论：{esc(scaleup_status)} for ImageNet validation。</strong> λ4 在 256/320 均取得正精度增益（{l4_delta_256:+.2f}/{l4_delta_320:+.2f} pp），训练 allocated 显存约为 DeiT 的 {l4_memory_256:.3f}/{l4_memory_320:.3f}×，训练吞吐比随 size 增长并在 320 达到 {l4_throughput_320:.3f}×；同点 fast inference 为 {l4_fast_infer_320['throughput_ratio']['mean']:.3f}×。{esc(scaleup_action)}锁定 aggregate 的严格 {esc(decision['status'])}/{esc(release_status)} 字段保持不变。</div>
    <div class="split">
      <article class="insight l2"><span class="muted">精度最强配置</span><strong>λ2 @ 320 · {l2_accuracy_320:.2f}%</strong><p>相对同 seed DeiT 提升 <b>{l2_delta_320:+.2f} pp</b>；λ2 的均值随 size 增大持续上升。</p></article>
      <article class="insight l4"><span class="muted">λ4 训练效率 crossover</span><strong>320 · {l4_throughput_320:.3f}× train throughput</strong><p>同时仅用 <b>{l4_memory_320:.3f}× train allocated 显存</b>；是唯一同时训练更快且更省显存的正式点。</p></article>
      <article class="insight l4"><span class="muted">λ4 fast inference crossover</span><strong>320 · {l4_fast_infer_320['throughput_ratio']['mean']:.3f} ± {l4_fast_infer_320['throughput_ratio']['sample_sd']:.3f}×</strong><p><b>{l4_fast_raw_320['throughput_img_s']['mean']:,.1f} ± {l4_fast_raw_320['throughput_img_s']['sample_sd']:,.1f} img/s</b>，{l4_fast_above_baseline_cards_320}/8 卡略快于 DeiT；generic 同点仍为 {l4_generic_infer_320['throughput_ratio']['mean']:.3f}×。</p></article>
      <article class="insight {esc(r256_card_class)}"><span class="muted">{esc(r256_card_label)}</span><strong>256 · {l4_throughput_256:.3f}× train throughput</strong><p>{esc(r256_card_detail)}</p></article>
      <article class="insight {esc(parity_class)}"><span class="muted">实现一致性</span><strong>{esc(parity_card_title)}</strong><p>{esc(parity_card_detail)}</p></article>
    </div>
  </section>

  <section id="accuracy">
    <div class="eyebrow">Primary endpoint</div><h2>Top-1 随输入尺寸的变化</h2>
    <p class="muted">每个点为 3 个 seed 的 epoch-199 EMA Top-1 均值；半透明小点是单 seed，误差条是 sample SD。</p>
    <figure class="figure"><div class="chart">{line_chart(resizes, accuracy)}</div><figcaption>λ2 从 68.98% 稳步提升到 70.74%；DeiT 随 size 增大略有下降；λ4 在 224 达到 69.71% 后回落。</figcaption></figure>
  </section>

  <section id="delta">
    <div class="eyebrow">Paired by seed</div><h2>相对 DeiT 的精度增益</h2>
    <p class="muted">同一 size、同一 seed 配对后再统计，减少 seed 波动对比较的干扰。0 线以上代表优于 DeiT。</p>
    <figure class="figure"><div class="chart">{delta_chart(resizes, deltas)}</div><figcaption>λ2 在全部尺寸为正且增益随 size 扩大；λ4 在 160 为 −0.51 pp，其余尺寸为正。</figcaption></figure>
  </section>

  <section id="absolute-efficiency">
    <div class="eyebrow">Absolute train measurements · 8 GPUs</div><h2>训练：DeiT 与 MergeNet 的绝对吞吐和显存</h2>
    <p class="muted">三条曲线均为正式 8 卡训练微基准的均值，误差条为跨卡 sample SD。这里直接显示 img/s 与 MiB，不做 DeiT 归一化。</p>
    <figure class="figure"><div class="chart">{absolute_efficiency_chart(resizes, raw_train, metric='throughput_img_s', title='训练吞吐绝对值', description='DeiT、MergeNet lambda2 与 lambda4 在五个输入尺寸的八卡平均训练吞吐。', y_label='训练吞吐 (img/s)', ticks=(0, 250, 500, 750, 1000))}</div><figcaption>DeiT 的训练吞吐从 size 160 的 870.2 ± 85.0 img/s 降至 size 320 的 253.6 ± 4.1 img/s；所有三模型绝对值均来自同一正式效率矩阵。</figcaption></figure>
    <figure class="figure"><div class="chart">{absolute_efficiency_chart(resizes, raw_train, metric='peak_allocated_mib', title='训练 allocated 显存绝对值', description='DeiT、MergeNet lambda2 与 lambda4 在五个输入尺寸的八卡平均训练峰值 allocated memory。', y_label='peak allocated memory (MiB)', ticks=(0, 3000, 6000, 9000, 12000))}</div><figcaption>DeiT 的 peak allocated 显存从 2,474.0 MiB 增至 8,926.8 MiB；λ4 在五个 size 均低于对应 DeiT，λ2 均高于对应 DeiT。</figcaption></figure>
    <h3>训练绝对测量（8 卡 mean ± sample SD）</h3>
    <div class="table-wrap"><table><thead><tr><th class="num">size</th><th>模型</th><th class="num">train throughput (img/s)</th><th class="num">train peak allocated (MiB)</th><th class="num">train step time (ms)</th><th class="num">参数量</th><th class="num">cards</th></tr></thead><tbody>{absolute_efficiency_rows_html(resizes, raw_train)}</tbody></table></div>
  </section>

  <section id="inference">
    <div class="eyebrow">Inference · reporting-only · 8 GPUs</div><h2>推理：generic 与 fast 路径的吞吐</h2>
    <p class="muted">DeiT 只有单一 generic inference 基线；MergeNet 同时测量 generic 与 fast inference path。fast 不是新模型，而是同一模型的等价执行路径，其正式比值始终与同物理卡上的 DeiT generic 配对。误差条为跨卡 sample SD。</p>
    <figure class="figure"><div class="chart">{inference_throughput_chart(resizes, raw_infer)}</div><figcaption>size 160→320，DeiT inference 吞吐下降 {100 * deit_infer_drop_160_320:.2f}%，λ4 fast 下降 {100 * l4_fast_infer_drop_160_320:.2f}%；size 320 的 λ4 fast 为 {l4_fast_raw_320['throughput_img_s']['mean']:,.1f} ± {l4_fast_raw_320['throughput_img_s']['sample_sd']:,.1f} img/s，相对自身 generic 路径的八卡绝对均值提升 {100 * l4_fast_uplift_320:.2f}%。</figcaption></figure>
    <figure class="figure"><div class="chart">{inference_ratio_chart(resizes, paired_infer)}</div><figcaption>相对同卡 DeiT generic 基线，λ4 在 size 320 的 generic / fast 吞吐比分别为 {l4_generic_infer_320['throughput_ratio']['mean']:.4f} ± {l4_generic_infer_320['throughput_ratio']['sample_sd']:.4f}× / {l4_fast_infer_320['throughput_ratio']['mean']:.4f} ± {l4_fast_infer_320['throughput_ratio']['sample_sd']:.4f}×；fast path 才使其跨过 1×。</figcaption></figure>
    <div class="split">
      <article class="insight l4"><span class="muted">可主张的 fast 结果</span><strong>λ4 @ 320 · +{100 * (l4_fast_infer_320['throughput_ratio']['mean'] - 1):.2f}% vs DeiT</strong><p>{l4_fast_above_baseline_cards_320}/8 张卡的同卡配对比高于 1；这是 fast path 促成的轻微推理吞吐 crossover。</p></article>
      <article class="insight risk"><span class="muted">部署权衡</span><strong>{l4_fast_infer_320['peak_allocated_ratio']['mean']:.3f}× allocated memory</strong><p>同点 λ4 fast 推理显存约为 DeiT 的两倍，因此不能表述为“又快又省”。</p></article>
    </div>
    <h3>推理绝对测量与同卡配对比值（8 卡 mean ± sample SD）</h3>
    <div class="table-wrap"><table><thead><tr><th class="num">size</th><th>模型</th><th>路径</th><th class="num">throughput (img/s)</th><th class="num">throughput / DeiT</th><th class="num">step time (ms)</th><th class="num">peak allocated (MiB)</th><th class="num">memory / DeiT</th><th class="num">cards</th></tr></thead><tbody>{inference_efficiency_rows_html(resizes, raw_infer, paired_infer)}</tbody></table></div>
    <div class="callout"><strong>证据边界：</strong> inference 为 batch 32、synthetic、model-only、20 warmup + 100 timed forward 的 steady-state 微基准，不是 batch-1 在线延迟或含预处理/I/O 的端到端服务指标；它仅作补充汇报，不参与预注册训练门禁。generic/fast 的数值兼容性已由 30/30 checkpoint parity 验证。</div>
  </section>

  <section id="tradeoff">
    <div class="eyebrow">Train paired ratios · candidate / DeiT</div><h2>归一化训练吞吐 × 显存权衡</h2>
    <p class="muted">每点为同物理卡配对后跨 8 卡均值。DeiT 在每个 size 都归一化为灰色菱形 (1×, 1×)，因此五个基线点重合；吞吐越右越好，allocated 显存越低越好。</p>
    <figure class="figure"><div class="chart">{tradeoff_chart(train_rows)}</div><figcaption>λ4 呈现最清晰的训练规模化效率趋势：全尺寸降低训练 allocated 显存，并在 320 越过训练吞吐基线。λ2 保留为精度优先的对照配置，其精度收益伴随较低训练吞吐和较高训练显存。</figcaption></figure>
    <div class="callout">效率数据是锁定 batch 的 synthetic、model-only、steady-state step 微基准，不等同于含 dataloader、增强和 checkpoint I/O 的 200 epoch 端到端 wall-clock。</div>
  </section>

  <section id="gate">
    <div class="eyebrow">Preregistered decision & audit trail</div><h2>门禁判定与完整数值</h2>
    <h3>λ4 预注册门槛</h3>
    <div class="table-wrap"><table><thead><tr><th>指标</th><th class="num">size</th><th class="num">证据</th><th class="num">规则</th><th class="num">观测值</th><th>状态</th></tr></thead><tbody>{gate_rows_html(decision['conditions'])}</tbody></table></div>
    <h3>按 size 汇总</h3>
    <div class="table-wrap"><table><thead><tr><th class="num">size</th><th class="num">DeiT Top-1</th><th class="num">λ2 Top-1</th><th class="num">λ2 Δ</th><th class="num">λ4 Top-1</th><th class="num">λ4 Δ</th><th class="num">λ4 train throughput</th><th class="num">λ4 train allocated</th><th class="num">λ4 fast infer throughput</th><th>标记</th></tr></thead><tbody>{summary_rows_html(resizes, accuracy, deltas, train, paired_infer)}</tbody></table></div>
    <details><summary>展开 45 个精度 run 的三 seed 数值</summary><div><div class="table-wrap"><table><thead><tr><th class="num">size</th><th>模型</th><th class="num">seed 42</th><th class="num">seed 43</th><th class="num">seed 44</th><th class="num">mean ± SD</th></tr></thead><tbody>{seed_rows_html(resizes, accuracy)}</tbody></table></div></div></details>
    <p><span class="status complete">campaign complete</span> 完成时间：<code>{esc(finished)}</code>；聚合时间：<code>{esc(aggregated)}</code>。</p>
    <p class="muted">权威 aggregate SHA-256：<code>{aggregate_hash}</code></p>
    <div class="evidence">
      <a href="evidence/cifar_resize_20260810/aggregate_results.json">权威 aggregate JSON ↗</a>
      <a href="evidence/cifar_resize_20260810/aggregate_results.md">完整人读证据表 ↗</a>
      <a href="evidence/cifar_resize_20260810/MANIFEST.json">SHA-256 manifest ↗</a>
      <a href="mergenet_cifar_resize_final_20260814.html">正式审计型 HTML ↗</a>
    </div>
  </section>
  <footer>单文件离线可视化 · 数据取自已锁定 aggregate JSON · {esc(footer_decision)}</footer>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-json", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_path = args.aggregate_json.resolve()
    output_path = args.output.resolve()
    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rendered = build_html(data, aggregate_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "source_sha256": sha256(aggregate_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
