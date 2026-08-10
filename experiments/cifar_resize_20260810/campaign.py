#!/usr/bin/env python3
"""Resumable eight-GPU scheduler for the preregistered CIFAR resize campaign.

The scheduler deliberately has no cluster-manager dependency.  Each physical GPU
obtains its own efficiency matrix before that card may enter the shared longest-
processing-time-first accuracy queue; cards pipeline independently.  Every launch
is preceded by two consecutive GPU idle probes, so a stale or externally occupied
device is never claimed blindly.

This file only orchestrates processes.  The benchmark and accuracy wrappers own
the model/training implementation and their on-disk artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import heapq
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = SCRIPT_DIR / "protocol.json"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return result


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON document without exposing a half-written state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_id(entry: Any) -> str:
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, Mapping):
        value = entry.get("id") or entry.get("model_id") or entry.get("name")
    else:
        value = None
    if not value or not isinstance(value, str):
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
    if not models or len(set(models)) != len(models):
        raise ValueError("protocol.models must contain unique model ids")
    if not resizes or len(set(resizes)) != len(resizes):
        raise ValueError("protocol.resizes must contain unique sizes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("protocol.seeds must contain unique seeds")
    expected = len(models) * len(resizes) * len(seeds)
    if expected != 45 and os.environ.get("ALLOW_NONSTANDARD_PLAN") != "1":
        raise ValueError(
            "the preregistered matrix must be 5 resizes x 3 models x 3 seeds "
            f"(45 jobs), but protocol expands to {expected}; set "
            "ALLOW_NONSTANDARD_PLAN=1 only for an intentional smoke test"
        )
    return models, resizes, seeds


def parse_gpus(raw: str) -> list[int]:
    pieces = [part.strip() for part in raw.split(",") if part.strip()]
    gpus = [_as_int(part, "GPU id") for part in pieces]
    if any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError(f"GPUS must be unique non-negative physical ids; got {raw!r}")
    if len(gpus) != 8 and os.environ.get("ALLOW_NONSTANDARD_GPU_COUNT") != "1":
        raise ValueError(
            f"this campaign requires exactly 8 physical GPUs; got {gpus}. "
            "ALLOW_NONSTANDARD_GPU_COUNT=1 is reserved for smoke tests"
        )
    return gpus


def nested_get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class Job:
    model: str
    resize: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}__r{self.resize}__s{self.seed}"


@dataclass
class Running:
    process: subprocess.Popen[str]
    gpu: int
    kind: str
    key: str
    command: list[str]
    log_path: Path
    log_handle: Any
    started_monotonic: float

    def close_log(self) -> None:
        try:
            self.log_handle.flush()
            self.log_handle.close()
        except Exception:
            pass


@dataclass(frozen=True)
class Paths:
    root: Path
    state_dir: Path
    runs_dir: Path
    benchmarks_dir: Path
    logs_dir: Path
    pids_dir: Path
    state_file: Path
    heartbeat_file: Path
    lock_file: Path
    master_pid_file: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        state_dir = root / "state"
        return cls(
            root=root,
            state_dir=state_dir,
            runs_dir=root / "runs",
            # ``efficiency/gpuN.json`` is the public per-card evidence layout.
            benchmarks_dir=root / "efficiency",
            logs_dir=root / "logs",
            pids_dir=state_dir / "pids",
            state_file=state_dir / "campaign_state.json",
            heartbeat_file=state_dir / "heartbeat.json",
            lock_file=state_dir / "campaign.lock",
            master_pid_file=state_dir / "master.pid",
        )

    def ensure(self) -> None:
        for path in (
            self.root,
            self.state_dir,
            self.runs_dir,
            self.benchmarks_dir,
            self.logs_dir,
            self.pids_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@contextmanager
def exclusive_master_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another campaign master holds {path}; refusing duplicate launch"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_utc={utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def job_dir(paths: Paths, job: Job) -> Path:
    # This path is part of the contract with run_accuracy_job.sh.
    return paths.runs_dir / job.model / f"r{job.resize}" / f"seed{job.seed}"


def _read_summary_target(summary: Path, target_epoch: int) -> tuple[bool, dict[str, Any]]:
    try:
        with summary.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return False, {"reason": f"unreadable summary: {exc}"}
    target_rows = []
    for row in rows:
        try:
            epoch = int(float(str(row.get("epoch", ""))))
        except (TypeError, ValueError):
            continue
        if epoch == target_epoch:
            target_rows.append(row)
    if not target_rows:
        return False, {"reason": f"summary lacks epoch {target_epoch}"}
    row = target_rows[-1]
    metric_key = next(
        (
            name
            for name in (
                "eval_top1_ema",
                "eval_ema_top1",
                "ema_top1",
                "eval_top1",
                "top1",
            )
            if row.get(name) not in (None, "")
        ),
        None,
    )
    if metric_key is None:
        return False, {"reason": "epoch row has no EMA/top1 metric"}
    try:
        metric = float(row[metric_key])
    except (TypeError, ValueError):
        return False, {"reason": f"invalid {metric_key}={row.get(metric_key)!r}"}
    if not math.isfinite(metric):
        return False, {"reason": f"non-finite {metric_key}"}
    return True, {
        "summary": str(summary),
        "epoch": target_epoch,
        "metric_key": metric_key,
        "ema_top1": metric,
    }


def _candidate_summary_files(directory: Path) -> list[Path]:
    preferred = [directory / "summary.csv", directory / "run" / "summary.csv"]
    found = [path for path in preferred if path.is_file()]
    if directory.is_dir():
        for path in directory.rglob("summary.csv"):
            if path not in found:
                found.append(path)
    return found


def _checkpoint_has_final_ema(checkpoint: Path, target_epoch: int) -> tuple[bool, str]:
    """Inspect a local trusted checkpoint; completion marker avoids this import."""

    metadata, error = _checkpoint_metadata(checkpoint)
    if metadata is None:
        return False, error
    try:
        epoch = int(metadata.get("epoch"))
    except (TypeError, ValueError):
        return False, "checkpoint has no integer epoch"
    has_ema = metadata.get("has_ema") is True
    if epoch != target_epoch:
        return False, f"checkpoint epoch={epoch}, expected {target_epoch}"
    if not has_ema:
        return False, "checkpoint lacks state_dict_ema/model_ema"
    return True, "ok"


def _checkpoint_metadata(checkpoint: Path) -> tuple[dict[str, Any] | None, str]:
    """Read epoch/EMA through the locked dependency root when campaign uses -S."""

    deps_root = os.environ.get("DEPS_ROOT")
    if deps_root:
        code = (
            "import json,sys,torch\n"
            "p=sys.argv[1]\n"
            "try:\n c=torch.load(p,map_location='cpu',weights_only=False)\n"
            "except TypeError:\n c=torch.load(p,map_location='cpu')\n"
            "print(json.dumps({'mapping':isinstance(c,dict),'epoch':c.get('epoch') if isinstance(c,dict) else None,"
            "'has_ema':isinstance(c,dict) and any(c.get(k) is not None for k in ('state_dict_ema','model_ema'))}))\n"
        )
        env = os.environ.copy()
        roots = [deps_root]
        if os.environ.get("RUNTIME_ROOT"):
            roots.append(os.environ["RUNTIME_ROOT"])
        env["PYTHONPATH"] = os.pathsep.join(roots)
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            result = subprocess.run(
                ["/usr/bin/python", "-S", "-c", code, str(checkpoint)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
            metadata = json.loads(result.stdout.strip().splitlines()[-1])
            if not metadata.get("mapping"):
                return None, "checkpoint is not a mapping"
            return metadata, "ok"
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
            return None, f"locked checkpoint inspection failed: {exc}"
    try:
        import torch
        try:
            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:
        return None, f"cannot read checkpoint: {exc}"
    if not isinstance(value, Mapping):
        return None, "checkpoint is not a mapping"
    return {
        "mapping": True,
        "epoch": value.get("epoch"),
        "has_ema": any(value.get(name) is not None for name in ("state_dict_ema", "model_ema")),
    }, "ok"


def inspect_job_progress_epoch(directory: Path, target_epoch: int) -> int | None:
    """Return a checkpoint/summary-agreed resumable epoch, otherwise None."""

    summary = next(iter(_candidate_summary_files(directory)), None)
    checkpoint = directory / "last.pth.tar"
    if summary is None or not checkpoint.is_file():
        return None
    try:
        with summary.open("r", encoding="utf-8", newline="") as handle:
            epochs = [
                int(float(row["epoch"]))
                for row in csv.DictReader(handle)
                if row.get("epoch") not in (None, "")
            ]
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        return None
    if not epochs:
        return None
    value, _ = _checkpoint_metadata(checkpoint)
    if value is None:
        return None
    try:
        checkpoint_epoch = int(value.get("epoch"))
    except (TypeError, ValueError):
        return None
    has_ema = value.get("has_ema") is True
    if (
        not has_ema
        or checkpoint_epoch < 0
        or checkpoint_epoch > target_epoch
        or max(epochs) != checkpoint_epoch
    ):
        return None
    return checkpoint_epoch


def accuracy_retry_transition(
    progress_before: int | None,
    progress_after: int | None,
    previous_failures: int,
    interrupted: bool,
    max_failures: int = 2,
) -> tuple[str, int, bool]:
    """Classify a non-complete launch without charging valid resume progress."""

    advanced = progress_after is not None and (
        progress_before is None or progress_after > progress_before
    )
    if interrupted:
        return "pending", previous_failures, advanced
    if advanced:
        return "pending", 0, True
    failures = previous_failures + 1
    return ("failed" if failures >= max_failures else "pending"), failures, False


def inspect_job_completion(directory: Path, target_epoch: int) -> tuple[bool, dict[str, Any]]:
    """Require epoch-199 summary plus a wrapper-verified or inspectable EMA checkpoint."""

    summaries = _candidate_summary_files(directory)
    summary_info: dict[str, Any] = {"reason": "summary.csv not found"}
    for summary in summaries:
        valid, info = _read_summary_target(summary, target_epoch)
        if valid:
            summary_info = info
            break
        summary_info = info
    else:
        return False, summary_info

    marker_candidates = [directory / "completion.json"]
    marker_candidates.extend(directory.glob("*/completion.json") if directory.is_dir() else [])
    for marker in marker_candidates:
        if not marker.is_file():
            continue
        try:
            value = load_json(marker)
            marker_epoch = int(value.get("epoch"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if (
            value.get("status") == "complete"
            and marker_epoch == target_epoch
            and value.get("ema") is True
        ):
            return True, {**summary_info, "verified_by": str(marker)}

    checkpoints: list[Path] = []
    for name in ("last.pth.tar", "last.pth", "checkpoint-last.pth.tar"):
        direct = directory / name
        if direct.is_file():
            checkpoints.append(direct)
    if directory.is_dir():
        checkpoints.extend(
            path
            for pattern in ("last.pth.tar", "last.pth", "checkpoint-last.pth.tar")
            for path in directory.rglob(pattern)
            if path not in checkpoints
        )
    reasons = []
    for checkpoint in checkpoints:
        valid, reason = _checkpoint_has_final_ema(checkpoint, target_epoch)
        if valid:
            return True, {**summary_info, "verified_by": str(checkpoint)}
        reasons.append(f"{checkpoint}: {reason}")
    return False, {
        **summary_info,
        "reason": "; ".join(reasons) if reasons else "final EMA checkpoint not found",
    }


def expected_benchmark_items() -> dict[str, tuple[str, int, str]]:
    expected: dict[str, tuple[str, int, str]] = {}
    modes = {
        "deit_s8": ("train", "infer"),
        "mn_l2": (
            "train_random_per_sample",
            "infer_generic",
            "infer_fast",
            "logits_parity",
        ),
        "mn_l4": (
            "train_random_per_sample",
            "infer_generic",
            "infer_fast",
            "logits_parity",
        ),
    }
    for resize in (160, 192, 224, 256, 320):
        for model, model_modes in modes.items():
            for mode in model_modes:
                item_id = f"{model}_r{resize}_{mode}"
                expected[item_id] = (model, resize, mode)
    return expected


def current_gpu_uuid(gpu: int) -> str | None:
    try:
        value = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    return value if value.startswith("GPU-") else None


def benchmark_is_complete(
    path: Path,
    expected_gpu: int | None = None,
    expected_gpu_uuid: str | None = None,
) -> bool:
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    header_ok = (
        isinstance(value, Mapping)
        and value.get("complete") is True
        and value.get("all_success") is True
    )
    if not header_ok:
        return False
    if expected_gpu is not None and expected_gpu_uuid is None:
        return False
    if value.get("canonical_environment") is not True or value.get("noncanonical") is True:
        return False
    if expected_gpu is not None:
        try:
            if int(value.get("physical_gpu")) != expected_gpu:
                return False
        except (TypeError, ValueError):
            return False
    if expected_gpu_uuid is not None and value.get("gpu_uuid") != expected_gpu_uuid:
        return False
    items = value.get("items")
    expected = expected_benchmark_items()
    if not isinstance(items, list) or len(items) != len(expected):
        return False
    observed_ids = [item.get("item_id") for item in items if isinstance(item, Mapping)]
    if len(observed_ids) != len(items) or len(set(observed_ids)) != len(observed_ids):
        return False
    if set(observed_ids) != set(expected):
        return False
    for item in items:
        if not isinstance(item, Mapping) or item.get("success") is not True:
            return False
        model, resize, mode = expected[str(item["item_id"])]
        try:
            identity_ok = (
                item.get("model_id") == model
                and int(item.get("resize")) == resize
                and item.get("mode") == mode
            )
        except (TypeError, ValueError):
            identity_ok = False
        if not identity_ok:
            return False
        item_environment = item.get("environment")
        if not isinstance(item_environment, Mapping):
            return False
        if item_environment.get("canonical") is not True or item_environment.get("noncanonical") is True:
            return False
        if expected_gpu is not None:
            try:
                if int(item_environment.get("physical_gpu")) != expected_gpu:
                    return False
            except (TypeError, ValueError):
                return False
        if (
            expected_gpu_uuid is not None
            and item_environment.get("gpu_uuid") != expected_gpu_uuid
        ):
            return False
        if mode == "logits_parity":
            if item.get("allclose") is not True:
                return False
        elif item.get("timing_valid") is not True:
            return False
    return True


def benchmark_waits_for_host_remeasure(
    path: Path,
    expected_gpu: int | None = None,
    expected_gpu_uuid: str | None = None,
) -> bool:
    """Distinguish invalid host timing from a functional benchmark failure."""

    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, Mapping):
        return False
    if (
        value.get("canonical_environment") is not True
        or value.get("noncanonical") is True
        or (expected_gpu is not None and expected_gpu_uuid is None)
    ):
        return False
    try:
        if expected_gpu is not None and int(value.get("physical_gpu")) != expected_gpu:
            return False
    except (TypeError, ValueError):
        return False
    if expected_gpu_uuid is not None and value.get("gpu_uuid") != expected_gpu_uuid:
        return False
    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        return False
    try:
        failed_items = int(summary.get("failed_items", -1))
        timing_invalid = int(summary.get("timing_invalid_items", 0))
    except (TypeError, ValueError):
        return False
    try:
        if int(summary.get("expected_items", -1)) != 50:
            return False
        if int(summary.get("missing_items", -1)) != 0:
            return False
    except (TypeError, ValueError):
        return False
    if failed_items != 0 or timing_invalid <= 0:
        return False
    expected = expected_benchmark_items()
    items = value.get("items")
    if not isinstance(items, list) or len(items) != len(expected):
        return False
    indexed = {item.get("item_id"): item for item in items if isinstance(item, Mapping)}
    if len(indexed) != len(items) or set(indexed) != set(expected):
        return False
    saw_invalid_timing = False
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
        ):
            return False
        if expected_gpu is not None:
            try:
                if int(environment.get("physical_gpu")) != expected_gpu:
                    return False
            except (TypeError, ValueError):
                return False
        if expected_gpu_uuid is not None and environment.get("gpu_uuid") != expected_gpu_uuid:
            return False
        if mode == "logits_parity":
            if item.get("allclose") is not True:
                return False
        elif item.get("timing_valid") is False:
            saw_invalid_timing = True
        elif item.get("timing_valid") is not True:
            return False
    return saw_invalid_timing


def effective_cpu_count() -> float:
    """Return the tighter of affinity and cgroup CPU quota."""

    try:
        affinity = float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity = float(os.cpu_count() or 1)
    quota_counts: list[float] = []
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text(encoding="utf-8").strip().split()[:2]
        if quota_text != "max":
            quota_counts.append(float(quota_text) / float(period_text))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            quota_counts.append(quota / period)
    except (OSError, ValueError):
        pass
    return max(0.01, min([affinity, *quota_counts]))


def gpu_probe(
    gpu: int,
    min_free_mib: int,
    max_util_pct: int,
    max_load1_per_cpu: float,
) -> dict[str, Any]:
    metric_cmd = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_cmd = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        metric = subprocess.run(
            metric_cmd, check=True, capture_output=True, text=True, timeout=15
        ).stdout.strip()
        first_line = metric.splitlines()[0]
        free_text, util_text = [part.strip() for part in first_line.split(",", 1)]
        free_mib = int(float(free_text))
        util_pct = int(float(util_text))
        apps_result = subprocess.run(
            app_cmd, check=True, capture_output=True, text=True, timeout=15
        )
        app_lines = []
        for line in apps_result.stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("no running"):
                continue
            first = stripped.split(",", 1)[0].strip()
            if first.isdigit():
                app_lines.append(stripped)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"ok": False, "reason": f"nvidia-smi probe failed: {exc}"}
    load1 = float(os.getloadavg()[0])
    cpus = effective_cpu_count()
    load1_per_cpu = load1 / cpus
    host_ok = load1_per_cpu <= max_load1_per_cpu
    ok = (
        not app_lines
        and free_mib >= min_free_mib
        and util_pct <= max_util_pct
        and host_ok
    )
    reason = "idle" if ok else (
        f"compute_apps={len(app_lines)}, free_mib={free_mib}/{min_free_mib}, "
        f"util_pct={util_pct}/{max_util_pct}, "
        f"load1_per_cpu={load1_per_cpu:.3f}/{max_load1_per_cpu:.3f}"
    )
    return {
        "ok": ok,
        "reason": reason,
        "free_mib": free_mib,
        "util_pct": util_pct,
        "compute_apps": app_lines,
        "load1": load1,
        "effective_cpus": cpus,
        "load1_per_cpu": load1_per_cpu,
        "max_load1_per_cpu": max_load1_per_cpu,
        "host_load_ok": host_ok,
        "checked_at": utc_now(),
    }


def benchmark_records(paths: Paths) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(paths.benchmarks_dir.glob("gpu*.json")):
        try:
            document = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        gpu = document.get("physical_gpu")
        try:
            expected_gpu = int(gpu)
        except (TypeError, ValueError):
            continue
        if not benchmark_is_complete(
            path,
            expected_gpu=expected_gpu,
            expected_gpu_uuid=current_gpu_uuid(expected_gpu),
        ):
            continue
        items = document.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if (
                not isinstance(item, Mapping)
                or not item.get("success")
                or (
                    item.get("mode") != "logits_parity"
                    and item.get("timing_valid") is not True
                )
            ):
                continue
            merged = dict(item)
            merged.setdefault("physical_gpu", gpu)
            records.append(merged)
    return records


def estimate_job_seconds(job: Job, records: Sequence[Mapping[str, Any]]) -> float:
    throughputs = []
    for row in records:
        try:
            same = (
                str(row.get("model_id")) == job.model
                and int(row.get("resize")) == job.resize
                and str(row.get("mode", "")).lower()
                in {"train", "training", "train_random_per_sample"}
            )
            value = float(row.get("throughput_img_s"))
        except (TypeError, ValueError):
            continue
        if same and math.isfinite(value) and value > 0:
            throughputs.append(value)
    if throughputs:
        # CIFAR-100 has 50k train images.  The common epoch multiplier does not
        # affect LPT ordering, but makes the estimate human-readable.
        return 50_000.0 * 200.0 / statistics.median(throughputs)
    # Before benchmarks exist (--dry-run), quadratic token count is the safest
    # neutral ordering.  A small model-family factor only breaks equal-size ties.
    name = job.model.lower()
    family_factor = (
        1.25
        if any(token in name for token in ("lambda2", "lam2", "_l2"))
        else 1.05
        if any(token in name for token in ("lambda4", "lam4", "_l4"))
        else 1.0
    )
    return float(job.resize * job.resize) * family_factor


class Campaign:
    def __init__(
        self,
        protocol: Mapping[str, Any],
        protocol_path: Path,
        paths: Paths,
        gpus: Sequence[int],
        benchmark_script: Path,
        accuracy_script: Path,
    ) -> None:
        self.protocol = protocol
        self.protocol_path = protocol_path
        self.paths = paths
        self.gpus = list(gpus)
        self.benchmark_script = benchmark_script
        self.accuracy_script = accuracy_script
        self.models, self.resizes, self.seeds = protocol_matrix(protocol)
        self.jobs = [
            Job(model, resize, seed)
            for resize in self.resizes
            for model in self.models
            for seed in self.seeds
        ]
        training = protocol.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("protocol.training must be an object")
        epochs = _as_int(training.get("epochs", 200), "training.epochs")
        self.target_epoch = epochs - 1
        if self.target_epoch != 199 and os.environ.get("ALLOW_NONSTANDARD_PLAN") != "1":
            raise ValueError(f"accuracy completion must be epoch 199; protocol requests {self.target_epoch}")
        efficiency = protocol.get("efficiency", {})
        if not isinstance(efficiency, Mapping):
            raise ValueError("protocol.efficiency must be an object")
        self.min_free_mib = _as_int(
            nested_get(
                efficiency,
                "min_free_mib",
                "min_free_mem_mib",
                "min_free_mem_mb",
                default=os.environ.get("GPU_MIN_FREE_MIB", 70_000),
            ),
            "efficiency.min_free_mib",
        )
        self.max_util_pct = _as_int(
            nested_get(
                efficiency,
                "max_idle_util_pct",
                "max_util_pct",
                default=os.environ.get("GPU_MAX_UTIL_PCT", 5),
            ),
            "efficiency.max_idle_util_pct",
        )
        self.max_load1_per_cpu = float(
            nested_get(
                efficiency,
                "max_load1_per_cpu",
                default=os.environ.get("MAX_LOAD1_PER_CPU", 1.5),
            )
        )
        if not math.isfinite(self.max_load1_per_cpu) or self.max_load1_per_cpu <= 0:
            raise ValueError("efficiency.max_load1_per_cpu must be a positive finite number")
        self.stability_checks = _as_int(
            nested_get(efficiency, "idle_consecutive_checks", default=2),
            "efficiency.idle_consecutive_checks",
        )
        if self.stability_checks < 2:
            raise ValueError("GPU launch requires at least two consecutive idle checks")
        self.stability_interval = float(
            nested_get(
                efficiency,
                "idle_check_interval_sec",
                default=os.environ.get("GPU_CHECK_INTERVAL_SEC", 10),
            )
        )
        self.poll_interval = float(os.environ.get("CAMPAIGN_POLL_SEC", "5"))
        self.max_attempts = 2  # initial attempt plus at most one retry
        self.stop_requested = False
        self.running: dict[int, Running] = {}
        self.state = self._initial_state()

    def _initial_state(self) -> dict[str, Any]:
        digest = protocol_digest(self.protocol)
        previous: dict[str, Any] = {}
        if self.paths.state_file.is_file():
            loaded = load_json(self.paths.state_file)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"invalid state document: {self.paths.state_file}")
            old_digest = loaded.get("protocol_sha256")
            if old_digest and old_digest != digest:
                raise RuntimeError(
                    "protocol.json changed after state creation; use a new CAMPAIGN_ROOT "
                    "instead of mixing preregistrations"
                )
            previous = loaded
        state = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.protocol.get("campaign_id", SCRIPT_DIR.name),
            "protocol": str(self.protocol_path),
            "protocol_sha256": digest,
            "master_pid": os.getpid(),
            "started_at": previous.get("started_at", utc_now()),
            "updated_at": utc_now(),
            "phase": previous.get("phase", "initializing"),
            "gpus": self.gpus,
            "gpu_probes": previous.get("gpu_probes", {}),
            "benchmarks": previous.get("benchmarks", {}),
            "jobs": previous.get("jobs", {}),
            "summary": previous.get("summary", {}),
        }
        for gpu in self.gpus:
            entry = state["benchmarks"].setdefault(
                str(gpu),
                {"status": "pending", "launches": 0, "functional_failures": 0},
            )
            entry.setdefault("launches", int(entry.get("attempts", 0)))
            entry.setdefault("functional_failures", 0)
            if benchmark_is_complete(
                self.paths.benchmarks_dir / f"gpu{gpu}.json",
                expected_gpu=gpu,
                expected_gpu_uuid=current_gpu_uuid(gpu),
            ):
                entry.update(status="completed", completed_at=utc_now())
            elif benchmark_waits_for_host_remeasure(
                self.paths.benchmarks_dir / f"gpu{gpu}.json",
                expected_gpu=gpu,
                expected_gpu_uuid=current_gpu_uuid(gpu),
            ):
                entry["status"] = "waiting_host_remeasure"
            elif entry.get("status") in {"running", "completed"}:
                entry["status"] = "pending"
        for job in self.jobs:
            entry = state["jobs"].setdefault(
                job.key,
                {
                    "model_id": job.model,
                    "resize": job.resize,
                    "seed": job.seed,
                    "status": "pending",
                    "launches": 0,
                    "no_progress_failures": 0,
                    "job_dir": str(job_dir(self.paths, job)),
                },
            )
            entry.setdefault("launches", int(entry.get("attempts", 0)))
            entry.setdefault("no_progress_failures", 0)
            complete, info = inspect_job_completion(job_dir(self.paths, job), self.target_epoch)
            if complete:
                entry.update(status="completed", completion=info, completed_at=utc_now())
            elif entry.get("status") == "running":
                progress = inspect_job_progress_epoch(job_dir(self.paths, job), self.target_epoch)
                entry.update(
                    status="pending",
                    last_error="master restarted during prior attempt; resumable state rechecked",
                    resumable_epoch=progress,
                )
        return state

    def write_state(self) -> None:
        counts: dict[str, int] = {}
        for entry in self.state["jobs"].values():
            status = str(entry.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        self.state["summary"] = {"accuracy_jobs": counts, "running_gpus": sorted(self.running)}
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.paths.state_file, self.state)
        atomic_write_json(
            self.paths.heartbeat_file,
            {
                "campaign_id": self.state["campaign_id"],
                "master_pid": os.getpid(),
                "phase": self.state["phase"],
                "running": {
                    str(gpu): {"kind": run.kind, "key": run.key, "pid": run.process.pid}
                    for gpu, run in self.running.items()
                },
                "at": self.state["updated_at"],
            },
        )

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "CAMPAIGN_ROOT": str(self.paths.root),
                "RUNS_ROOT": str(self.paths.runs_dir),
                "PROTOCOL_PATH": str(self.protocol_path),
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
                "OPENTOME_MERGENET_IMPL": "new",
                "TIMM_FUSED_ATTN": "1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("PYTHONOPTIMIZE", None)
        for name in (
            "CUDA_LAUNCH_BLOCKING",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        ):
            env.pop(name, None)
        for name in (
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(name, None)
        for name in list(env):
            if name.startswith("TORCHELASTIC_"):
                env.pop(name, None)
        # Preserve explicit deployment paths verbatim.  Wrappers perform their
        # own existence/version checks because they know the expected layout.
        for name in ("RUNTIME_ROOT", "DEPS_ROOT", "DATA_DIR"):
            if name in os.environ:
                env[name] = os.environ[name]
        return env

    def _launch(
        self,
        gpu: int,
        kind: str,
        key: str,
        command: list[str],
        log_path: Path,
        env: Mapping[str, str],
    ) -> Running:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(
            f"\n[{utc_now()}] launch gpu={gpu} kind={kind} key={key}\n"
            f"command={json.dumps(command)}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=str(SCRIPT_DIR),
            env=dict(env),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        running = Running(
            process=process,
            gpu=gpu,
            kind=kind,
            key=key,
            command=command,
            log_path=log_path,
            log_handle=log_handle,
            started_monotonic=time.monotonic(),
        )
        self.running[gpu] = running
        atomic_write_json(
            self.paths.pids_dir / f"gpu{gpu}.json",
            {
                "pid": process.pid,
                "gpu": gpu,
                "kind": kind,
                "key": key,
                "log": str(log_path),
                "started_at": utc_now(),
            },
        )
        return running

    def probe(self, gpu: int, streaks: dict[int, int]) -> bool:
        result = gpu_probe(
            gpu,
            self.min_free_mib,
            self.max_util_pct,
            self.max_load1_per_cpu,
        )
        streaks[gpu] = streaks.get(gpu, 0) + 1 if result.get("ok") else 0
        result["consecutive_ok"] = streaks[gpu]
        result["required_consecutive_ok"] = self.stability_checks
        self.state["gpu_probes"][str(gpu)] = result
        return streaks[gpu] >= self.stability_checks

    def benchmark_command(self, gpu: int, output: Path) -> list[str]:
        efficiency = self.protocol.get("efficiency", {})
        command = [
            sys.executable,
            "-S",
            str(self.benchmark_script),
            "--gpu",
            str(gpu),
            "--out",
            str(output),
        ]
        options = (
            ("batch_size", "--batch-size"),
            ("warmup", "--warmup"),
            ("steps", "--steps"),
        )
        for key, flag in options:
            value = efficiency.get(key) if isinstance(efficiency, Mapping) else None
            if value is not None:
                command.extend([flag, str(value)])
        return command

    def run_benchmarks(self) -> None:
        self.state["phase"] = "benchmark"
        streaks = {gpu: 0 for gpu in self.gpus}
        last_probe = {gpu: 0.0 for gpu in self.gpus}
        self.write_state()
        while True:
            pending = [
                gpu
                for gpu in self.gpus
                if self.state["benchmarks"][str(gpu)].get("status")
                not in {"completed", "failed"}
                and gpu not in self.running
            ]
            if not pending and not self.running:
                break
            now = time.monotonic()
            for gpu in pending:
                if now - last_probe[gpu] < self.stability_interval:
                    continue
                last_probe[gpu] = now
                if not self.probe(gpu, streaks):
                    continue
                entry = self.state["benchmarks"][str(gpu)]
                attempts = int(entry.get("attempts", 0)) + 1
                if attempts > self.max_attempts:
                    entry.update(status="failed", last_error="benchmark retries exhausted")
                    continue
                output = self.paths.benchmarks_dir / f"gpu{gpu}.json"
                log = self.paths.logs_dir / "benchmarks" / f"gpu{gpu}.attempt{attempts}.log"
                command = self.benchmark_command(gpu, output)
                env = self._base_env()
                env.update({"PHYSICAL_GPU": str(gpu)})
                run = self._launch(gpu, "benchmark", f"gpu{gpu}", command, log, env)
                entry.update(
                    status="running",
                    attempts=attempts,
                    pid=run.process.pid,
                    log=str(log),
                    output=str(output),
                    started_at=utc_now(),
                )
                streaks[gpu] = 0
            for gpu, run in list(self.running.items()):
                returncode = run.process.poll()
                if returncode is None:
                    continue
                run.close_log()
                self.running.pop(gpu)
                pid_file = self.paths.pids_dir / f"gpu{gpu}.json"
                if pid_file.exists():
                    pid_file.unlink()
                entry = self.state["benchmarks"][str(gpu)]
                output = self.paths.benchmarks_dir / f"gpu{gpu}.json"
                if returncode == 0 and benchmark_is_complete(
                    output,
                    expected_gpu=gpu,
                    expected_gpu_uuid=current_gpu_uuid(gpu),
                ):
                    entry.update(status="completed", completed_at=utc_now(), returncode=returncode)
                elif int(entry.get("attempts", 0)) < self.max_attempts:
                    entry.update(
                        status="pending",
                        returncode=returncode,
                        last_error="benchmark exited without a complete matrix",
                    )
                else:
                    entry.update(
                        status="failed",
                        returncode=returncode,
                        last_error="benchmark retries exhausted",
                    )
                streaks[gpu] = 0
                last_probe[gpu] = time.monotonic()
            self.write_state()
            if self.stop_requested:
                raise KeyboardInterrupt
            time.sleep(self.poll_interval)
        failed = [
            gpu
            for gpu in self.gpus
            if self.state["benchmarks"][str(gpu)].get("status") != "completed"
        ]
        if failed:
            raise RuntimeError(
                f"efficiency matrix incomplete on GPUs {failed}; accuracy queue was not started"
            )

    def accuracy_command(self, job: Job) -> list[str]:
        return [
            "bash",
            str(self.accuracy_script),
            job.model,
            str(job.resize),
            str(job.seed),
            # The physical GPU id is the fourth positional wrapper argument.
            "{GPU}",
        ]

    def run_accuracy(self) -> None:
        self.state["phase"] = "accuracy"
        records = benchmark_records(self.paths)
        heap: list[tuple[float, str, Job]] = []
        for job in self.jobs:
            entry = self.state["jobs"][job.key]
            complete, info = inspect_job_completion(job_dir(self.paths, job), self.target_epoch)
            if complete:
                entry.update(status="completed", completion=info, completed_at=utc_now())
                continue
            if int(entry.get("attempts", 0)) >= self.max_attempts:
                entry["status"] = "failed"
                continue
            entry["status"] = "pending"
            estimate = estimate_job_seconds(job, records)
            entry["lpt_estimated_seconds"] = estimate
            heapq.heappush(heap, (-estimate, job.key, job))

        streaks = {gpu: 0 for gpu in self.gpus}
        last_probe = {gpu: 0.0 for gpu in self.gpus}
        self.write_state()
        while heap or self.running:
            now = time.monotonic()
            idle_gpus = [gpu for gpu in self.gpus if gpu not in self.running]
            for gpu in idle_gpus:
                if not heap or now - last_probe[gpu] < self.stability_interval:
                    continue
                last_probe[gpu] = now
                if not self.probe(gpu, streaks):
                    continue
                _, _, job = heapq.heappop(heap)
                entry = self.state["jobs"][job.key]
                attempts = int(entry.get("attempts", 0)) + 1
                if attempts > self.max_attempts:
                    entry.update(status="failed", last_error="accuracy retries exhausted")
                    continue
                directory = job_dir(self.paths, job)
                directory.mkdir(parents=True, exist_ok=True)
                log = self.paths.logs_dir / "accuracy" / job.key / f"attempt{attempts}.log"
                command = self.accuracy_command(job)
                command[-1] = str(gpu)
                env = self._base_env()
                env.update(
                    {
                        "JOB_DIR": str(directory),
                        "MODEL_ID": job.model,
                        "RESIZE": str(job.resize),
                        "SEED": str(job.seed),
                        "PHYSICAL_GPU": str(gpu),
                    }
                )
                run = self._launch(gpu, "accuracy", job.key, command, log, env)
                entry.update(
                    status="running",
                    attempts=attempts,
                    gpu=gpu,
                    pid=run.process.pid,
                    log=str(log),
                    started_at=utc_now(),
                )
                streaks[gpu] = 0

            for gpu, run in list(self.running.items()):
                returncode = run.process.poll()
                if returncode is None:
                    continue
                run.close_log()
                self.running.pop(gpu)
                pid_file = self.paths.pids_dir / f"gpu{gpu}.json"
                if pid_file.exists():
                    pid_file.unlink()
                entry = self.state["jobs"][run.key]
                job = Job(str(entry["model_id"]), int(entry["resize"]), int(entry["seed"]))
                complete, info = inspect_job_completion(job_dir(self.paths, job), self.target_epoch)
                entry["returncode"] = returncode
                if complete:
                    entry.update(status="completed", completion=info, completed_at=utc_now())
                elif int(entry.get("attempts", 0)) < self.max_attempts:
                    entry.update(
                        status="pending",
                        last_error=(
                            f"wrapper rc={returncode}; final epoch/EMA checkpoint not verified: "
                            f"{info.get('reason', 'unknown')}"
                        ),
                    )
                    estimate = float(entry.get("lpt_estimated_seconds", 0.0))
                    heapq.heappush(heap, (-estimate, job.key, job))
                else:
                    entry.update(
                        status="failed",
                        last_error=(
                            f"accuracy retries exhausted (last rc={returncode}): "
                            f"{info.get('reason', 'final artifact missing')}"
                        ),
                    )
                streaks[gpu] = 0
                last_probe[gpu] = time.monotonic()
            self.write_state()
            if self.stop_requested:
                raise KeyboardInterrupt
            time.sleep(self.poll_interval)
        failed = [key for key, value in self.state["jobs"].items() if value.get("status") != "completed"]
        self.state["phase"] = "complete" if not failed else "completed_with_failures"
        self.state["finished_at"] = utc_now()
        self.write_state()
        if failed:
            raise RuntimeError(f"{len(failed)} accuracy jobs exhausted their retry; see state/logs")

    def run_interleaved(self, benchmark_only: bool = False) -> None:
        """Pipeline each card from benchmark into the shared accuracy queue.

        A busy card may remain in the idle gate indefinitely, but it does not hold
        back cards whose own efficiency matrix is already complete.  LPT scores
        are rebuilt from every newly completed matrix before each dispatch.
        """

        self.state["phase"] = "benchmark_and_accuracy" if not benchmark_only else "benchmark"
        pending_jobs: dict[str, Job] = {}
        if not benchmark_only:
            for job in self.jobs:
                entry = self.state["jobs"][job.key]
                complete, info = inspect_job_completion(job_dir(self.paths, job), self.target_epoch)
                if complete:
                    entry.update(status="completed", completion=info, completed_at=utc_now())
                elif int(entry.get("no_progress_failures", 0)) >= self.max_attempts:
                    entry["status"] = "failed"
                else:
                    entry["status"] = "pending"
                    pending_jobs[job.key] = job

        streaks = {gpu: 0 for gpu in self.gpus}
        last_probe = {gpu: 0.0 for gpu in self.gpus}
        self.write_state()

        while True:
            benchmark_terminal = all(
                self.state["benchmarks"][str(gpu)].get("status") in {"completed", "failed"}
                for gpu in self.gpus
            )
            if not self.running:
                if benchmark_only and benchmark_terminal:
                    break
                if not benchmark_only and benchmark_terminal and not pending_jobs:
                    break
                ready_cards = [
                    gpu
                    for gpu in self.gpus
                    if self.state["benchmarks"][str(gpu)].get("status") == "completed"
                ]
                if not benchmark_only and benchmark_terminal and pending_jobs and not ready_cards:
                    break

            now = time.monotonic()
            for gpu in self.gpus:
                if gpu in self.running or now - last_probe[gpu] < self.stability_interval:
                    continue
                benchmark_entry = self.state["benchmarks"][str(gpu)]
                benchmark_status = benchmark_entry.get("status")
                wants_benchmark = benchmark_status in {"pending", "waiting_host_remeasure"}
                wants_accuracy = (
                    not benchmark_only
                    and benchmark_status == "completed"
                    and bool(pending_jobs)
                )
                if not wants_benchmark and not wants_accuracy:
                    continue
                last_probe[gpu] = now
                if not self.probe(gpu, streaks):
                    continue

                if wants_benchmark:
                    functional_failures = int(benchmark_entry.get("functional_failures", 0))
                    if functional_failures >= self.max_attempts:
                        benchmark_entry.update(status="failed", last_error="benchmark retries exhausted")
                        continue
                    launches = int(benchmark_entry.get("launches", 0)) + 1
                    output = self.paths.benchmarks_dir / f"gpu{gpu}.json"
                    log = self.paths.logs_dir / "benchmarks" / f"gpu{gpu}.launch{launches}.log"
                    command = self.benchmark_command(gpu, output)
                    env = self._base_env()
                    env.update({"PHYSICAL_GPU": str(gpu)})
                    run = self._launch(gpu, "benchmark", f"gpu{gpu}", command, log, env)
                    benchmark_entry.update(
                        status="running",
                        launches=launches,
                        functional_failures=functional_failures,
                        pid=run.process.pid,
                        log=str(log),
                        output=str(output),
                        started_at=utc_now(),
                    )
                else:
                    # Rebuild immediately before every dispatch: newly completed
                    # card measurements can change the median runtime ordering.
                    records = benchmark_records(self.paths)
                    lpt_heap = [
                        (-estimate_job_seconds(job, records), job.key, job)
                        for job in pending_jobs.values()
                    ]
                    heapq.heapify(lpt_heap)
                    neg_estimate, _, job = heapq.heappop(lpt_heap)
                    pending_jobs.pop(job.key)
                    entry = self.state["jobs"][job.key]
                    failures = int(entry.get("no_progress_failures", 0))
                    if failures >= self.max_attempts:
                        entry.update(status="failed", last_error="accuracy no-progress retries exhausted")
                        continue
                    directory = job_dir(self.paths, job)
                    directory.mkdir(parents=True, exist_ok=True)
                    launches = int(entry.get("launches", 0)) + 1
                    progress_before = inspect_job_progress_epoch(directory, self.target_epoch)
                    log = self.paths.logs_dir / "accuracy" / job.key / f"launch{launches}.log"
                    command = self.accuracy_command(job)
                    command[-1] = str(gpu)
                    env = self._base_env()
                    env.update(
                        {
                            "JOB_DIR": str(directory),
                            "MODEL_ID": job.model,
                            "RESIZE": str(job.resize),
                            "SEED": str(job.seed),
                            "PHYSICAL_GPU": str(gpu),
                        }
                    )
                    run = self._launch(gpu, "accuracy", job.key, command, log, env)
                    entry.update(
                        status="running",
                        launches=launches,
                        no_progress_failures=failures,
                        progress_before_epoch=progress_before,
                        gpu=gpu,
                        pid=run.process.pid,
                        log=str(log),
                        lpt_estimated_seconds=-neg_estimate,
                        started_at=utc_now(),
                    )
                streaks[gpu] = 0

            for gpu, run in list(self.running.items()):
                returncode = run.process.poll()
                if returncode is None:
                    continue
                run.close_log()
                self.running.pop(gpu)
                pid_file = self.paths.pids_dir / f"gpu{gpu}.json"
                if pid_file.exists():
                    pid_file.unlink()
                if run.kind == "benchmark":
                    entry = self.state["benchmarks"][str(gpu)]
                    output = self.paths.benchmarks_dir / f"gpu{gpu}.json"
                    if returncode == 0 and benchmark_is_complete(
                        output,
                        expected_gpu=gpu,
                        expected_gpu_uuid=current_gpu_uuid(gpu),
                    ):
                        entry.update(status="completed", completed_at=utc_now(), returncode=returncode)
                    elif benchmark_waits_for_host_remeasure(
                        output,
                        expected_gpu=gpu,
                        expected_gpu_uuid=current_gpu_uuid(gpu),
                    ):
                        entry.update(
                            status="waiting_host_remeasure",
                            returncode=returncode,
                            last_error=(
                                "functional items passed but host timing gate was violated; "
                                "waiting for an idle-host relaunch without consuming retry"
                            ),
                        )
                    else:
                        failures = int(entry.get("functional_failures", 0)) + 1
                        entry["functional_failures"] = failures
                        entry["returncode"] = returncode
                        if failures < self.max_attempts:
                            entry.update(
                                status="pending",
                                last_error="functional benchmark failure; one retry remains",
                            )
                        else:
                            entry.update(
                                status="failed",
                                last_error="functional benchmark retry exhausted before all_success",
                            )
                else:
                    entry = self.state["jobs"][run.key]
                    job = Job(str(entry["model_id"]), int(entry["resize"]), int(entry["seed"]))
                    complete, info = inspect_job_completion(job_dir(self.paths, job), self.target_epoch)
                    entry["returncode"] = returncode
                    if complete:
                        entry.update(status="completed", completion=info, completed_at=utc_now())
                    else:
                        directory = job_dir(self.paths, job)
                        progress_before = entry.get("progress_before_epoch")
                        progress_after = inspect_job_progress_epoch(directory, self.target_epoch)
                        interrupted = self.stop_requested or returncode in {130, -signal.SIGINT}
                        retry_status, failures, advanced = accuracy_retry_transition(
                            None if progress_before is None else int(progress_before),
                            progress_after,
                            int(entry.get("no_progress_failures", 0)),
                            interrupted,
                            self.max_attempts,
                        )
                        entry["no_progress_failures"] = failures
                        entry["resumable_epoch"] = progress_after
                        if interrupted or advanced:
                            entry.update(
                                status=retry_status,
                                last_error=(
                                    "interrupted with resumable state; failure budget unchanged"
                                    if interrupted
                                    else f"partial checkpoint advanced to epoch {progress_after}; resuming"
                                ),
                            )
                            pending_jobs[job.key] = job
                            streaks[gpu] = 0
                            last_probe[gpu] = time.monotonic()
                            continue
                        if retry_status == "pending":
                            entry.update(
                                status="pending",
                                last_error=(
                                    f"wrapper rc={returncode}; no checkpoint progress "
                                    f"({progress_before!r}->{progress_after!r}); one retry remains; "
                                    f"completion={info.get('reason', 'unknown')}"
                                ),
                            )
                            pending_jobs[job.key] = job
                        else:
                            entry.update(
                                status="failed",
                                last_error=(
                                    f"two consecutive no-progress failures (last rc={returncode}, "
                                    f"epoch={progress_after!r}): "
                                    f"{info.get('reason', 'final artifact missing')}"
                                ),
                            )
                streaks[gpu] = 0
                last_probe[gpu] = time.monotonic()

            self.write_state()
            if self.stop_requested:
                raise KeyboardInterrupt
            time.sleep(self.poll_interval)

        benchmark_failed = [
            gpu
            for gpu in self.gpus
            if self.state["benchmarks"][str(gpu)].get("status") != "completed"
        ]
        accuracy_failed = (
            []
            if benchmark_only
            else [
                key
                for key, value in self.state["jobs"].items()
                if value.get("status") != "completed"
            ]
        )
        self.state["phase"] = (
            "complete" if not benchmark_failed and not accuracy_failed else "completed_with_failures"
        )
        self.state["finished_at"] = utc_now()
        self.write_state()
        if benchmark_failed or accuracy_failed:
            raise RuntimeError(
                f"campaign incomplete: benchmark GPUs={benchmark_failed}, "
                f"accuracy jobs={len(accuracy_failed)}; see state/logs"
            )

    def request_stop(self, signum: int, _frame: Any) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.state["phase"] = "stopping"
        self.state["stop_signal"] = signum
        self.write_state()
        # Forward SIGINT to each process group so the trainer can take its normal
        # KeyboardInterrupt/checkpoint path.  State remains resumable afterward.
        for run in self.running.values():
            try:
                os.killpg(run.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


def dry_plan(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    root: Path,
    gpus: Sequence[int],
    benchmark_script: Path,
    accuracy_script: Path,
) -> dict[str, Any]:
    models, resizes, seeds = protocol_matrix(protocol)
    jobs = [Job(model, resize, seed) for resize in resizes for model in models for seed in seeds]
    paths = Paths.from_root(root)
    estimates = benchmark_records(paths) if paths.benchmarks_dir.is_dir() else []
    ordered = sorted(jobs, key=lambda job: (-estimate_job_seconds(job, estimates), job.key))
    efficiency = protocol.get("efficiency", {})
    sample_gpu = gpus[0] if gpus else 0
    sample_benchmark = [
        sys.executable,
        "-S",
        str(benchmark_script),
        "--gpu",
        str(sample_gpu),
        "--out",
        str(paths.benchmarks_dir / f"gpu{sample_gpu}.json"),
    ]
    for key, flag in (("batch_size", "--batch-size"), ("warmup", "--warmup"), ("steps", "--steps")):
        if isinstance(efficiency, Mapping) and efficiency.get(key) is not None:
            sample_benchmark.extend([flag, str(efficiency[key])])
    return {
        "dry_run": True,
        "campaign_id": protocol.get("campaign_id", SCRIPT_DIR.name),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_digest(protocol),
        "campaign_root": str(root),
        "physical_gpus": list(gpus),
        "matrix": {
            "models": models,
            "resizes": resizes,
            "seeds": seeds,
            "job_count": len(jobs),
        },
        "launch_gate": {
            "no_compute_apps": True,
            "consecutive_checks": max(2, _as_int(nested_get(efficiency, "idle_consecutive_checks", default=2), "idle checks")),
            "min_free_mib": _as_int(nested_get(efficiency, "min_free_mib", "min_free_mem_mib", "min_free_mem_mb", default=70_000), "min free"),
            "max_util_pct": _as_int(nested_get(efficiency, "max_idle_util_pct", "max_util_pct", default=5), "max util"),
            "max_load1_per_cpu": float(
                nested_get(efficiency, "max_load1_per_cpu", default=1.5)
            ),
        },
        "benchmark_command_example": sample_benchmark,
        "accuracy_command_example": [
            "bash",
            str(accuracy_script),
            ordered[0].model,
            str(ordered[0].resize),
            str(ordered[0].seed),
            str(sample_gpu),
        ],
        "lpt_order": [
            {
                "job": job.key,
                "estimated_seconds_or_relative_cost": estimate_job_seconds(job, estimates),
                "job_dir": str(job_dir(paths, job)),
            }
            for job in ordered
        ],
        "will_execute": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(os.environ.get("PROTOCOL_PATH", DEFAULT_PROTOCOL)),
        help="preregistered protocol.json",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(os.environ.get("CAMPAIGN_ROOT", SCRIPT_DIR)),
        help="runtime root for state, logs, benchmarks, and runs",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the 45-job plan; launch nothing")
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="stop after all eight per-card benchmark matrices are complete",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol.expanduser().resolve()
    root = args.campaign_root.expanduser().resolve()
    protocol = load_json(protocol_path)
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol root must be a JSON object")
    gpus = parse_gpus(os.environ.get("GPUS", "0,1,2,3,4,5,6,7"))
    benchmark_script = Path(
        os.environ.get("BENCHMARK_SCRIPT", SCRIPT_DIR / "benchmark_resize.py")
    ).expanduser().resolve()
    accuracy_script = Path(
        os.environ.get("RUN_ACCURACY_SCRIPT", SCRIPT_DIR / "run_accuracy_job.sh")
    ).expanduser().resolve()
    if args.dry_run:
        print(
            json.dumps(
                dry_plan(protocol, protocol_path, root, gpus, benchmark_script, accuracy_script),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for required in (benchmark_script, accuracy_script):
        if not required.is_file():
            raise FileNotFoundError(required)
    paths = Paths.from_root(root)
    paths.ensure()
    with exclusive_master_lock(paths.lock_file):
        atomic_write_json(
            paths.master_pid_file,
            {"pid": os.getpid(), "host": os.uname().nodename, "started_at": utc_now()},
        )
        campaign = Campaign(
            protocol=protocol,
            protocol_path=protocol_path,
            paths=paths,
            gpus=gpus,
            benchmark_script=benchmark_script,
            accuracy_script=accuracy_script,
        )
        signal.signal(signal.SIGINT, campaign.request_stop)
        signal.signal(signal.SIGTERM, campaign.request_stop)
        try:
            campaign.run_interleaved(benchmark_only=args.benchmark_only)
        except KeyboardInterrupt:
            campaign.state["phase"] = "interrupted_resumable"
            campaign.write_state()
            return 130
        except Exception as exc:
            campaign.state["phase"] = "error"
            campaign.state["last_error"] = repr(exc)
            campaign.write_state()
            raise
        finally:
            for run in campaign.running.values():
                run.close_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
