#!/usr/bin/env python3
"""Read-only campaign, GPU, host-gate, and accuracy progress monitor."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(
    "/liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/"
    "cifar_resize_delivery_validation_20260810"
)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def latest_summary(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / "summary.csv"
    if not path.is_file():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return {"path": str(path), "error": "unreadable"}
    if not rows:
        return {"path": str(path), "rows": 0}
    row = rows[-1]
    result: dict[str, Any] = {"path": str(path), "rows": len(rows)}
    for source, target, cast in (
        ("epoch", "epoch", lambda value: int(float(value))),
        ("eval_top1", "ema_top1", float),
        ("train_loss", "train_loss", float),
        ("train_effective_lambda", "effective_lambda", float),
    ):
        try:
            if row.get(source) not in (None, ""):
                result[target] = cast(row[source])
        except (TypeError, ValueError):
            result[target] = row.get(source)
    return result


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(raw) != 6:
            continue
        index, uuid, name, used, free, util = (value.strip() for value in raw)
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "used_mib": int(float(used)),
                "free_mib": int(float(free)),
                "util_pct": int(float(util)),
                "compute_pids": [],
            }
        )
    by_uuid = {row["uuid"]: row for row in rows}
    app_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        apps = subprocess.run(
            app_command, check=True, capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        apps = ""
    for raw in csv.reader(apps.splitlines(), skipinitialspace=True):
        if len(raw) != 4:
            continue
        uuid, pid, process_name, used = (value.strip() for value in raw)
        if uuid in by_uuid and pid.isdigit():
            by_uuid[uuid]["compute_pids"].append(
                {"pid": int(pid), "name": process_name, "used_mib": used}
            )
    return rows


def snapshot(root: Path) -> dict[str, Any]:
    state = load_json(root / "state" / "campaign_state.json") or {}
    heartbeat = load_json(root / "state" / "heartbeat.json") or {}
    master = load_json(root / "state" / "master.pid") or {}
    jobs = state.get("jobs", {}) if isinstance(state, Mapping) else {}
    benchmarks = state.get("benchmarks", {}) if isinstance(state, Mapping) else {}

    job_counts: dict[str, int] = {}
    running: list[dict[str, Any]] = []
    completed_metrics: list[dict[str, Any]] = []
    for key, raw_entry in jobs.items():
        entry = raw_entry if isinstance(raw_entry, Mapping) else {}
        status = str(entry.get("status", "unknown"))
        job_counts[status] = job_counts.get(status, 0) + 1
        directory = Path(entry.get("job_dir", root / "runs"))
        progress = latest_summary(directory)
        item = {
            "key": key,
            "status": status,
            "gpu": entry.get("gpu"),
            "pid": entry.get("pid"),
            "attempts": entry.get("attempts", 0),
            "log": entry.get("log"),
            "progress": progress,
        }
        if status == "running":
            running.append(item)
        if status == "completed" and progress:
            completed_metrics.append(item)

    benchmark_counts: dict[str, int] = {}
    for raw_entry in benchmarks.values():
        entry = raw_entry if isinstance(raw_entry, Mapping) else {}
        status = str(entry.get("status", "unknown"))
        benchmark_counts[status] = benchmark_counts.get(status, 0) + 1

    affinity_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    load1 = os.getloadavg()[0]
    host = {
        "load1": load1,
        "affinity_cpu_count": affinity_cpus,
        "load1_per_cpu": load1 / affinity_cpus if affinity_cpus else None,
        "gate_threshold": 1.5,
    }
    host["gate_ok"] = host["load1_per_cpu"] is not None and host["load1_per_cpu"] <= 1.5

    now = dt.datetime.now(dt.timezone.utc)
    heartbeat_at = parse_timestamp(heartbeat.get("at"))
    return {
        "observed_at": now.isoformat(timespec="seconds"),
        "campaign_root": str(root),
        "phase": state.get("phase", "not_started"),
        "master": {
            "pid": master.get("pid"),
            "alive": process_alive(master.get("pid")),
            "heartbeat_at": heartbeat.get("at"),
            "heartbeat_age_sec": (
                round((now - heartbeat_at.astimezone(dt.timezone.utc)).total_seconds(), 1)
                if heartbeat_at else None
            ),
        },
        "host": host,
        "job_counts": job_counts,
        "benchmark_counts": benchmark_counts,
        "running": sorted(running, key=lambda item: (item.get("gpu") is None, item.get("gpu", 999))),
        "completed_with_metrics": len(completed_metrics),
        "gpus": query_gpus(),
        "state_updated_at": state.get("updated_at"),
        "last_error": state.get("last_error"),
    }


def compact_counts(values: Mapping[str, int]) -> str:
    if not values:
        return "none"
    preferred = ("completed", "running", "pending", "failed")
    keys = [key for key in preferred if key in values]
    keys.extend(sorted(set(values) - set(keys)))
    return " ".join(f"{key}={values[key]}" for key in keys)


def render(document: Mapping[str, Any]) -> str:
    master = document["master"]
    host = document["host"]
    lines = [
        f"[{document['observed_at']}] phase={document['phase']} root={document['campaign_root']}",
        (
            f"master pid={master.get('pid')} alive={master.get('alive')} "
            f"heartbeat_age={master.get('heartbeat_age_sec')}s"
        ),
        (
            f"host load1={host['load1']:.2f} cpus={host['affinity_cpu_count']} "
            f"load/cpu={host['load1_per_cpu']:.3f} gate<=1.5:{host['gate_ok']}"
        ),
        f"benchmarks: {compact_counts(document['benchmark_counts'])}",
        f"accuracy:   {compact_counts(document['job_counts'])}",
    ]
    if document.get("last_error"):
        lines.append(f"last_error: {document['last_error']}")
    lines.append("GPU  util  used/free MiB  apps  assigned")
    assignments = {item.get("gpu"): item for item in document["running"]}
    for gpu in document["gpus"]:
        if "error" in gpu:
            lines.append(f"GPU query error: {gpu['error']}")
            continue
        assigned = assignments.get(gpu["index"])
        label = assigned["key"] if assigned else "-"
        progress = assigned.get("progress") if assigned else None
        if progress and progress.get("epoch") is not None:
            label += f" ep={progress['epoch']}/199 top1={progress.get('ema_top1', '-')}"
        lines.append(
            f"{gpu['index']:>3}  {gpu['util_pct']:>3}%  "
            f"{gpu['used_mib']:>5}/{gpu['free_mib']:<5}  "
            f"{len(gpu['compute_pids']):>4}  {label}"
        )
    for item in document["running"]:
        if item.get("gpu") in {gpu.get("index") for gpu in document["gpus"]}:
            continue
        lines.append(f"assigned(nonvisible GPU): {item['key']} gpu={item.get('gpu')}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(os.environ.get("CAMPAIGN_ROOT", DEFAULT_ROOT)),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--watch", action="store_true", help="refresh until interrupted")
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 1:
        raise SystemExit("--interval must be >= 1 second")
    root = args.campaign_root.expanduser().resolve()
    while True:
        document = snapshot(root)
        if args.json:
            print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(render(document))
        if not args.watch:
            return 0
        print()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
