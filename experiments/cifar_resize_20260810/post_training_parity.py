#!/usr/bin/env python3
"""Evaluate generic/fast parity on all trained MergeNet EMA checkpoints.

This is a post-training release gate, not part of the accuracy/efficiency
campaign scheduler.  The formal matrix contains 30 tasks:

    (mn_l2, mn_l4) x (160, 192, 224, 256, 320) x (42, 43, 44)

Each task evaluates the same epoch-199 EMA checkpoint and the same full,
deterministic CIFAR-100 test loader with ``alternating_per_layer`` followed by
``alternating_per_layer_fast`` for every batch.  Results are independently
locked and atomically written, so the matrix can be resumed or sharded across
physical GPUs without mixing partial JSON documents.

Only the Python standard library is imported until the requested physical GPU
has been mapped to an NVIDIA UUID and the exact runtime/dependency roots have
been installed at the front of ``sys.path``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "mergenet.cifar_resize_checkpoint_parity.v1"
RUNNER_REVISION = 1
MODEL_IDS: Tuple[str, ...] = ("mn_l2", "mn_l4")
RESIZES: Tuple[int, ...] = (160, 192, 224, 256, 320)
SEEDS: Tuple[int, ...] = (42, 43, 44)
NUM_CLASSES = 100
EXPECTED_SAMPLES = 10_000
TOP1_DELTA_LIMIT_PP = 0.05
TOP1_MISMATCH_LIMIT = 5
GENERIC_GROUPING = "alternating_per_layer"
FAST_GROUPING = "alternating_per_layer_fast"
CIFAR100_TEST_MD5 = "f0ef6b0ae62326f3e7ffdfab6717acfc"
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_CAMPAIGN_ROOT = Path(
    "/liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/"
    "cifar_resize_delivery_validation_20260810"
)
DEFAULT_DEPS_ROOT = Path("/liziqing/yukai/.deps_mergenet_resize20260810")

EXPECTED_ENVIRONMENT: Mapping[str, str] = {
    "python": "3.10",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "timm": "0.9.11",
    "flash_attn": "2.7.4.post1",
}
EXPECTED_RUNTIME_ENV: Mapping[str, str] = {
    "OPENTOME_MERGENET_IMPL": "new",
    "TIMM_FUSED_ATTN": "1",
}
FORBIDDEN_DISTRIBUTED_ENV = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)
FORBIDDEN_PERTURBATION_ENV = (
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)

EXPECTED_MODEL_GEOMETRY: Mapping[str, Mapping[str, Any]] = {
    "mn_l2": {
        "model": "mergenet_small_cls",
        "patch_size": 8,
        "local_depth": 6,
        "latent_depth": 6,
        "lambda_start": 2,
        "lambda_local": 2,
        "local_block_window": 16,
        "dtem_window_size": 8,
    },
    "mn_l4": {
        "model": "mergenet_small_cls",
        "patch_size": 8,
        "local_depth": 4,
        "latent_depth": 8,
        "lambda_start": 2,
        "lambda_local": 4,
        "local_block_window": 32,
        "dtem_window_size": 8,
    },
}


@dataclasses.dataclass(frozen=True)
class Task:
    model_id: str
    resize: int
    seed: int
    validation_batch_size: int

    @property
    def task_id(self) -> str:
        return f"{self.model_id}__r{self.resize}__s{self.seed}"

    @property
    def filename(self) -> str:
        return f"{self.model_id}_r{self.resize}_seed{self.seed}.json"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "resize": self.resize,
            "seed": self.seed,
            "validation_batch_size": self.validation_batch_size,
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class TaskLock:
    def __init__(self, output: Path):
        self.path = Path(f"{output}.lock")
        self._handle: Optional[Any] = None

    def __enter__(self) -> "TaskLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise RuntimeError(f"refusing symlinked task lock: {self.path}")
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "unknown owner"
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"task output is locked: {self.path} ({owner})") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(
            json.dumps(
                {"pid": os.getpid(), "host": socket.gethostname(), "at": utc_now()},
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def physical_gpu(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("GPU must be a non-negative decimal index")
    if value != "0" and value.startswith("0"):
        raise argparse.ArgumentTypeError("GPU index must not contain leading zeroes")
    return int(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(os.environ.get("CAMPAIGN_ROOT", DEFAULT_CAMPAIGN_ROOT)),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="default: CAMPAIGN_ROOT/runtime/cifar_resize_20260810/protocol.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="default: CAMPAIGN_ROOT/runtime/imagenet_longtrain_v1",
    )
    parser.add_argument(
        "--deps-root",
        type=Path,
        default=Path(os.environ.get("DEPS_ROOT", DEFAULT_DEPS_ROOT)),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/liziqing/yukai/data")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: CAMPAIGN_ROOT/post_training_parity",
    )
    parser.add_argument(
        "--gpu",
        type=physical_gpu,
        help="physical nvidia-smi index; required unless --dry-run",
    )
    parser.add_argument("--workers", type=nonnegative_int, default=8)
    parser.add_argument("--log-interval", type=positive_int, default=20)
    parser.add_argument("--only-model", action="append", choices=MODEL_IDS)
    parser.add_argument("--only-resize", action="append", type=int, choices=RESIZES)
    parser.add_argument("--only-seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--shard-count", type=positive_int, default=1)
    parser.add_argument("--shard-index", type=nonnegative_int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit protocol/snapshot/checkpoint readiness without torch, CUDA, locks, or writes",
    )
    args = parser.parse_args(argv)
    if args.shard_index >= args.shard_count:
        parser.error("--shard-index must be smaller than --shard-count")
    if not args.dry_run and args.gpu is None:
        parser.error("--gpu is required for a formal evaluation")
    return args


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(
        protocol.get("schema_version") == "mergenet.cifar_resize_protocol.v1",
        "unexpected protocol schema_version",
    )
    require(
        protocol.get("expected_environment") == dict(EXPECTED_ENVIRONMENT),
        "protocol expected_environment differs from the release lock",
    )
    require(
        protocol.get("expected_runtime_env") == dict(EXPECTED_RUNTIME_ENV),
        "protocol expected_runtime_env differs from the release lock",
    )
    models = protocol.get("models")
    require(isinstance(models, list), "protocol.models must be a list")
    model_index = {
        entry.get("id"): entry
        for entry in models
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
    }
    require(
        [entry.get("id") for entry in models if isinstance(entry, Mapping)]
        == ["deit_s8", "mn_l2", "mn_l4"],
        "protocol model order/matrix drift",
    )
    for model_id, geometry in EXPECTED_MODEL_GEOMETRY.items():
        require(model_id in model_index, f"protocol missing {model_id}")
        require(
            model_index[model_id].get("kind") == "mergenet",
            f"protocol {model_id} is not marked mergenet",
        )
        require(
            model_index[model_id].get("geometry") == dict(geometry),
            f"protocol {model_id} geometry differs from the release lock",
        )
    expected_resizes = [
        (160, 200),
        (192, 200),
        (224, 200),
        (256, 100),
        (320, 50),
    ]
    actual_resizes = []
    for entry in protocol.get("resizes", []):
        require(isinstance(entry, Mapping), "protocol resize entry must be an object")
        actual_resizes.append((int(entry.get("size", -1)), int(entry.get("micro_batch", -1))))
    require(actual_resizes == expected_resizes, "protocol resize/batch matrix drift")
    require(protocol.get("seeds") == list(SEEDS), "protocol seed matrix drift")

    training = protocol.get("training")
    require(isinstance(training, Mapping), "protocol.training must be an object")
    locked_training = {
        "dataset": "CIFAR100",
        "dataset_download": False,
        "num_classes": 100,
        "epochs": 200,
        "target_epoch": 199,
        "amp": "fp16",
        "drop_rate": 0,
        "attention_drop_rate": 0,
        "drop_path_rate": 0.1,
        "validation_crop_pct": 0.9,
        "model_ema": True,
        "prefetcher": False,
        "pin_memory": True,
        "workers": 8,
    }
    for key, expected in locked_training.items():
        require(training.get(key) == expected, f"protocol training.{key} drift")
    expected_mn = {
        "train_grouping": "random_per_sample",
        "train_grouping_seed": 0,
        "eval_grouping": FAST_GROUPING,
        "eval_grouping_seed": 0,
        "dtem_feat_dim": 64,
        "dtem_r": 2,
        "dtem_t": 1,
        "metric_grad_scale": 0.1,
        "source_trace_mode": "center",
        "total_merge_latent": 0,
        "use_softkmax": True,
        "swa_size": 256,
        "local_cls_global": True,
        "lambda_ramp_start_epoch": 0,
        "lambda_ramp_epochs": 50,
        "soft_topk": True,
        "soft_topk_aux_weight": 0.05,
        "soft_topk_aux_start_epoch": 20,
        "soft_topk_aux_ramp_epochs": 20,
    }
    require(training.get("mergenet") == expected_mn, "protocol MergeNet knobs drift")

    expected_gate = {
        "blocks_campaign_launch": False,
        "required_for_final_release": True,
        "scope": "30 个 MergeNet epoch-199 EMA checkpoints（2 MN models × 5 resizes × 3 seeds）",
        "dataset": "CIFAR-100 test 10000 images",
        "protocol": "同一 checkpoint、同一完整 deterministic loader 分别运行 alternating_per_layer 与 alternating_per_layer_fast",
        "record": [
            "generic_top1",
            "fast_top1",
            "top1_delta_pp",
            "argmax_mismatch_count",
            "max_abs_logit_diff",
            "mean_abs_logit_diff",
        ],
        "per_run_condition": "abs(top1_delta_pp) <= 0.05（最多 5/10000；top-1 分辨率 0.01pp）",
        "failure_policy": "任一 run 失败则 release NO-GO；mandatory 性能 gate 数值仍原样报告，不得删除或改写",
    }
    require(
        protocol.get("post_training_release_gate") == expected_gate,
        "protocol post_training_release_gate differs from the preregistration",
    )


def build_tasks(protocol: Mapping[str, Any]) -> List[Task]:
    resize_batches = {
        int(entry["size"]): int(entry["micro_batch"])
        for entry in protocol["resizes"]
    }
    return [
        Task(model_id, resize, seed, resize_batches[resize])
        for resize in RESIZES
        for model_id in MODEL_IDS
        for seed in SEEDS
    ]


def select_tasks(tasks: Sequence[Task], args: argparse.Namespace) -> List[Task]:
    models = set(args.only_model or MODEL_IDS)
    resizes = set(args.only_resize or RESIZES)
    seeds = set(args.only_seed or SEEDS)
    selected = [
        task
        for index, task in enumerate(tasks)
        if task.model_id in models
        and task.resize in resizes
        and task.seed in seeds
        and index % args.shard_count == args.shard_index
    ]
    if not selected:
        raise ValueError("filters/shard selected no parity tasks")
    return selected


def validate_no_symlink(path: Path, root: Path, label: str) -> None:
    root = root.resolve(strict=True)
    candidate = path.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes campaign root: {path}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} contains symlink component: {cursor}")


def tree_hashes(root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"immutable runtime contains symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = hash_file(path)
        elif not path.is_dir():
            raise ValueError(f"immutable runtime contains unsupported entry: {path}")
    return result


def validate_campaign_snapshot(
    campaign_root: Path,
    runtime_root: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_runtime = campaign_root / "runtime" / "imagenet_longtrain_v1"
    expected_protocol = campaign_root / "runtime" / "cifar_resize_20260810" / "protocol.json"
    require(runtime_root == expected_runtime.resolve(strict=True), "formal runtime-root must be the campaign immutable snapshot")
    require(protocol_path == expected_protocol.resolve(strict=True), "formal protocol must be the campaign immutable snapshot protocol")
    snapshot_path = campaign_root / "runtime" / "snapshot_manifest.json"
    state_path = campaign_root / "state" / "campaign_state.json"
    for path, label in ((snapshot_path, "snapshot manifest"), (state_path, "campaign state")):
        validate_no_symlink(path, campaign_root, label)
        require(path.is_file() and not path.is_symlink(), f"{label} missing or non-regular: {path}")
    snapshot = load_json(snapshot_path)
    require(isinstance(snapshot, Mapping), "snapshot manifest must be an object")
    expected_runtime_files = snapshot.get("runtime_files")
    expected_harness_files = snapshot.get("harness_files")
    require(isinstance(expected_runtime_files, Mapping), "snapshot runtime_files missing")
    require(isinstance(expected_harness_files, Mapping), "snapshot harness_files missing")
    actual_runtime_files = tree_hashes(runtime_root)
    require(actual_runtime_files == dict(expected_runtime_files), "immutable runtime failed file-hash verification")
    protocol_file_sha = hash_file(protocol_path)
    require(
        expected_harness_files.get("protocol.json") == protocol_file_sha,
        "protocol file does not match immutable snapshot manifest",
    )
    runtime_fingerprint = sha256_bytes(canonical_json(actual_runtime_files))
    protocol_canonical_sha = sha256_bytes(canonical_json(protocol))
    state = load_json(state_path)
    require(isinstance(state, Mapping), "campaign state must be an object")
    require(
        state.get("protocol_sha256") == protocol_canonical_sha,
        "campaign state protocol digest mismatch",
    )
    require(
        state.get("campaign_id") == protocol.get("campaign_id"),
        "campaign state id mismatch",
    )
    return {
        "snapshot_manifest_path": str(snapshot_path.resolve()),
        "snapshot_manifest_sha256": hash_file(snapshot_path),
        "snapshot_bundle_sha256": snapshot.get("bundle_sha256"),
        "runtime_tree_sha256": runtime_fingerprint,
        "runtime_files": actual_runtime_files,
        "protocol_file_sha256": protocol_file_sha,
        "protocol_canonical_sha256": protocol_canonical_sha,
        "campaign_state_path": str(state_path.resolve()),
        "campaign_state_sha256": hash_file(state_path),
        "campaign_phase": state.get("phase"),
        "campaign_jobs": state.get("jobs", {}),
    }


def validate_data(data_dir: Path) -> Dict[str, Any]:
    test_path = data_dir / "cifar-100-python" / "test"
    require(test_path.is_file() and not test_path.is_symlink(), f"CIFAR-100 test file missing: {test_path}")
    observed_md5 = md5_file(test_path)
    require(observed_md5 == CIFAR100_TEST_MD5, "CIFAR-100 test MD5 differs from the torchvision canonical file")
    return {
        "root": str(data_dir),
        "test_path": str(test_path.resolve()),
        "test_size_bytes": test_path.stat().st_size,
        "test_md5": observed_md5,
        "test_sha256": hash_file(test_path),
        "expected_samples": EXPECTED_SAMPLES,
    }


def canonical_job_dir(root: Path, task: Task) -> Path:
    return root / "runs" / task.model_id / f"r{task.resize}" / f"seed{task.seed}"


def checkpoint_evidence_stdlib(
    campaign_root: Path,
    task: Task,
    campaign: Mapping[str, Any],
) -> Dict[str, Any]:
    job_dir = canonical_job_dir(campaign_root, task)
    marker_path = job_dir / "completion.json"
    checkpoint_path = job_dir / "last.pth.tar"
    summary_path = job_dir / "summary.csv"
    for path, label in (
        (job_dir, "job directory"),
        (marker_path, "completion marker"),
        (checkpoint_path, "checkpoint"),
        (summary_path, "summary"),
    ):
        validate_no_symlink(path, campaign_root, label)
    require(job_dir.is_dir(), f"job directory missing: {job_dir}")
    require(marker_path.is_file() and not marker_path.is_symlink(), f"completion marker missing: {marker_path}")
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), f"checkpoint missing: {checkpoint_path}")
    require(summary_path.is_file() and not summary_path.is_symlink(), f"summary missing: {summary_path}")
    marker = load_json(marker_path)
    require(isinstance(marker, Mapping), "completion marker must be an object")
    expected_marker = {
        "status": "complete",
        "epoch": 199,
        "ema": True,
        "model_id": task.model_id,
        "resize": task.resize,
        "seed": task.seed,
    }
    for key, expected in expected_marker.items():
        require(marker.get(key) == expected, f"completion marker {key} mismatch for {task.task_id}")
    require(Path(str(marker.get("checkpoint_path", ""))).resolve() == checkpoint_path.resolve(), "completion checkpoint_path mismatch")
    require(Path(str(marker.get("summary_path", ""))).resolve() == summary_path.resolve(), "completion summary_path mismatch")
    checkpoint_sha = hash_file(checkpoint_path)
    summary_sha = hash_file(summary_path)
    require(marker.get("checkpoint_sha256") == checkpoint_sha, "completion checkpoint SHA-256 mismatch")
    require(marker.get("summary_sha256") == summary_sha, "completion summary SHA-256 mismatch")

    manifest_path = Path(str(marker.get("manifest_path", "")))
    validate_no_symlink(manifest_path, campaign_root, "accuracy manifest")
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "accuracy manifest missing or non-regular")
    manifest = load_json(manifest_path)
    require(isinstance(manifest, Mapping), "accuracy manifest must be an object")
    manifest_job = manifest.get("job")
    require(isinstance(manifest_job, Mapping), "accuracy manifest job missing")
    require(manifest_job.get("model_id") == task.model_id, "accuracy manifest model mismatch")
    require(int(manifest_job.get("resize", -1)) == task.resize, "accuracy manifest resize mismatch")
    require(int(manifest_job.get("seed", -1)) == task.seed, "accuracy manifest seed mismatch")
    require(int(manifest_job.get("target_epoch", -1)) == 199, "accuracy manifest target epoch mismatch")
    manifest_hashes = manifest.get("hashes")
    require(isinstance(manifest_hashes, Mapping), "accuracy manifest hashes missing")
    require(
        manifest_hashes.get("protocol_sha256") == campaign["protocol_file_sha256"],
        "accuracy manifest protocol SHA mismatch",
    )
    require(
        manifest_hashes.get("runtime_tree_sha256") == campaign["runtime_tree_sha256"],
        "accuracy manifest runtime tree SHA mismatch",
    )
    state_jobs = campaign.get("campaign_jobs")
    require(isinstance(state_jobs, Mapping), "campaign state jobs missing")
    state_job = state_jobs.get(task.task_id)
    require(isinstance(state_job, Mapping), f"campaign state lacks {task.task_id}")
    require(state_job.get("status") == "completed", f"campaign state does not mark {task.task_id} completed")

    # The summary is independently checked at the preregistered endpoint.  Its
    # fast-mode EMA value is supplemental; the parity runner never substitutes
    # a best epoch for epoch 199.
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_rows = []
    for row in rows:
        try:
            if int(float(row.get("epoch", -1))) == 199:
                target_rows.append(row)
        except (TypeError, ValueError):
            continue
    require(target_rows, "summary.csv lacks epoch 199")
    summary_top1 = float(target_rows[-1].get("eval_top1", "nan"))
    require(math.isfinite(summary_top1), "summary epoch-199 eval_top1 is non-finite")
    marker_top1 = float(marker.get("ema_top1", "nan"))
    require(math.isfinite(marker_top1) and marker_top1 == summary_top1, "completion EMA top1 mismatch")
    return {
        "job_dir": str(job_dir.resolve()),
        "completion_path": str(marker_path.resolve()),
        "completion_sha256": hash_file(marker_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": summary_sha,
        "summary_epoch199_fast_ema_top1": summary_top1,
        "accuracy_manifest_path": str(manifest_path.resolve()),
        "accuracy_manifest_sha256": hash_file(manifest_path),
    }


def query_physical_gpus() -> Dict[int, Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot enumerate physical GPUs: {exc}") from exc
    rows: Dict[int, Dict[str, Any]] = {}
    for fields in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi row: {fields!r}")
        index_text, uuid, name, pci_bus_id, memory_mib, driver = (
            field.strip() for field in fields
        )
        index = int(index_text)
        rows[index] = {
            "physical_index": index,
            "uuid": uuid,
            "name": name,
            "pci_bus_id": pci_bus_id,
            "total_memory_mib": int(memory_mib),
            "driver_version": driver,
        }
    require(bool(rows), "nvidia-smi returned no GPUs")
    return rows


def map_physical_gpu(index: int) -> Dict[str, Any]:
    gpus = query_physical_gpus()
    require(index in gpus, f"physical GPU {index} not found; available={sorted(gpus)}")
    gpu = dict(gpus[index])
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(
        inherited in (None, "", str(index), gpu["uuid"]),
        f"incompatible inherited CUDA_VISIBLE_DEVICES={inherited!r}",
    )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    gpu["inherited_cuda_visible_devices"] = inherited
    gpu["exported_cuda_visible_devices"] = gpu["uuid"]
    gpu["process_cuda_index"] = 0
    return gpu


def validate_process_environment(runtime_root: Path, deps_root: Path) -> None:
    require(sys.flags.optimize == 0, "optimized Python (-O/PYTHONOPTIMIZE) is forbidden")
    require("PYTHONOPTIMIZE" not in os.environ, "PYTHONOPTIMIZE must be unset")
    for name in FORBIDDEN_DISTRIBUTED_ENV:
        require(name not in os.environ, f"inherited distributed variable {name} is forbidden")
    torchelastic = sorted(name for name in os.environ if name.startswith("TORCHELASTIC_"))
    require(not torchelastic, f"inherited torch elastic variables are forbidden: {torchelastic}")
    for name in FORBIDDEN_PERTURBATION_ENV:
        require(name not in os.environ, f"inherited CUDA/PyTorch perturbation {name} is forbidden")
    for name, expected in EXPECTED_RUNTIME_ENV.items():
        observed = os.environ.get(name)
        require(observed in (None, expected), f"{name} must equal {expected}")
        os.environ[name] = expected
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    raw_pythonpath = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    resolved_pythonpath = [str(Path(entry).expanduser().resolve()) for entry in raw_pythonpath]
    expected_pythonpath = [str(runtime_root), str(deps_root)]
    require(
        resolved_pythonpath == expected_pythonpath,
        "formal PYTHONPATH must contain exactly RUNTIME_ROOT:DEPS_ROOT in that order; "
        f"actual={resolved_pythonpath!r}",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(expected_pythonpath)
    filtered_sys_path = []
    for entry in sys.path:
        if not entry:
            filtered_sys_path.append(entry)
            continue
        resolved = str(Path(entry).expanduser().resolve())
        if resolved in expected_pythonpath:
            continue
        if "site-packages" in resolved or "dist-packages" in resolved:
            continue
        filtered_sys_path.append(entry)
    sys.path[:] = expected_pythonpath + filtered_sys_path


def module_under(module: Any, root: Path) -> bool:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        return False
    try:
        Path(module_path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_exact_runtime(
    runtime_root: Path,
    deps_root: Path,
    gpu: Mapping[str, Any],
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    import torch
    import torchvision
    import timm
    import flash_attn
    import opentome
    import opentome.models.mergenet.model as mergenet_model

    modules = {
        "torch": torch,
        "torchvision": torchvision,
        "timm": timm,
        "flash_attn": flash_attn,
    }
    for name, module in modules.items():
        require(module_under(module, deps_root), f"{name} imported outside exact DEPS_ROOT: {getattr(module, '__file__', None)}")
    require(module_under(opentome, runtime_root), f"opentome imported outside exact RUNTIME_ROOT: {opentome.__file__}")
    require(module_under(mergenet_model, runtime_root), "MergeNet model imported outside exact RUNTIME_ROOT")
    actual = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "flash_attn": getattr(flash_attn, "__version__", distribution_version("flash-attn")),
    }
    require(actual == dict(EXPECTED_ENVIRONMENT), f"exact dependency mismatch: actual={actual!r}")
    require(torch.cuda.is_available(), "CUDA unavailable after UUID mapping")
    require(torch.cuda.device_count() == 1, "formal runner must see exactly one CUDA device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    require(properties.name.strip() == str(gpu["name"]).strip(), "torch/nvidia-smi GPU name mismatch")
    torch.backends.cudnn.benchmark = True
    environment = {
        "canonical": True,
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "versions": actual,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "runtime_env": {name: os.environ.get(name) for name in EXPECTED_RUNTIME_ENV},
        "pythonpath": os.environ.get("PYTHONPATH"),
        "gpu": dict(gpu),
        "torch_gpu": {
            "process_index": 0,
            "name": properties.name,
            "total_memory_mib": int(properties.total_memory // (1024 ** 2)),
            "compute_capability": [properties.major, properties.minor],
        },
        "modules": {
            "torch": str(Path(torch.__file__).resolve()),
            "torchvision": str(Path(torchvision.__file__).resolve()),
            "timm": str(Path(timm.__file__).resolve()),
            "flash_attn": str(Path(flash_attn.__file__).resolve()),
            "opentome": str(Path(opentome.__file__).resolve()),
            "mergenet_model": str(Path(mergenet_model.__file__).resolve()),
        },
    }
    environment["fingerprint"] = sha256_bytes(
        canonical_json(
            {
                "versions": environment["versions"],
                "cuda_runtime": environment["cuda_runtime"],
                "cudnn": environment["cudnn"],
                "runtime_env": environment["runtime_env"],
                "pythonpath": environment["pythonpath"],
                "gpu": environment["gpu"],
                "modules": environment["modules"],
            }
        )
    )
    return torch, torchvision, timm, environment


def model_kwargs(task: Task, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    model_entry = next(entry for entry in protocol["models"] if entry["id"] == task.model_id)
    geometry = model_entry["geometry"]
    training = protocol["training"]
    mn = training["mergenet"]
    return {
        "pretrained": False,
        "num_classes": NUM_CLASSES,
        "img_size": task.resize,
        "patch_size": geometry["patch_size"],
        "local_depth": geometry["local_depth"],
        "latent_depth": geometry["latent_depth"],
        "lambda_local": float(geometry["lambda_local"]),
        "total_merge_latent": mn["total_merge_latent"],
        "dtem_window_size": geometry["dtem_window_size"],
        "dtem_feat_dim": mn["dtem_feat_dim"],
        "dtem_r": mn["dtem_r"],
        "dtem_t": mn["dtem_t"],
        "dtem_train_grouping": mn["train_grouping"],
        "dtem_train_grouping_seed": mn["train_grouping_seed"],
        "dtem_eval_grouping": FAST_GROUPING,
        "dtem_eval_grouping_seed": mn["eval_grouping_seed"],
        "use_softkmax": mn["use_softkmax"],
        "metric_grad_scale": mn["metric_grad_scale"],
        "source_trace_mode": mn["source_trace_mode"],
        "swa_size": mn["swa_size"],
        "local_block_window": geometry["local_block_window"],
        "local_cls_global": mn["local_cls_global"],
        "soft_topk": mn["soft_topk"],
        "soft_topk_aux_weight": mn["soft_topk_aux_weight"],
        "drop_rate": training["drop_rate"],
        "attn_drop_rate": training["attention_drop_rate"],
        "drop_path_rate": training["drop_path_rate"],
    }


def extract_logits(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("model returned an empty tuple/list")
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output", "pred"):
            if key in output:
                return output[key]
        raise TypeError(f"unsupported model output keys: {sorted(output)}")
    return output


def checkpoint_arg(checkpoint: Mapping[str, Any], name: str) -> Any:
    args = checkpoint.get("args")
    if isinstance(args, Mapping):
        return args.get(name)
    return getattr(args, name, None)


def load_ema_checkpoint(
    task: Task,
    evidence: Dict[str, Any],
    model: Any,
    torch: Any,
) -> None:
    checkpoint_path = evidence["checkpoint_path"]
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require(
        hash_file(Path(checkpoint_path)) == evidence["checkpoint_sha256"],
        "checkpoint changed while it was being loaded",
    )
    require(isinstance(checkpoint, Mapping), "checkpoint is not a mapping")
    require(int(checkpoint.get("epoch", -1)) == 199, "checkpoint epoch is not 199")
    # The delivery trainer/timm 0.9.11 writes this exact key.  Do not silently
    # fall back to live ``state_dict`` or a differently-shaped legacy EMA
    # representation: this gate is specifically about the epoch-199 EMA model.
    state_key = "state_dict_ema"
    state_dict = checkpoint.get(state_key)
    require(isinstance(state_dict, Mapping), "checkpoint lacks mapping state_dict_ema")
    from timm.models._helpers import clean_state_dict

    cleaned = clean_state_dict(state_dict)
    incompatible = model.load_state_dict(cleaned, strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, "strict EMA load returned incompatible keys")
    geometry = EXPECTED_MODEL_GEOMETRY[task.model_id]
    locked_args = {
        "model": "mergenet_small_cls",
        "img_size": task.resize,
        "patch_size": 8,
        "num_classes": 100,
        "seed": task.seed,
        "model_ema": True,
        "local_depth": geometry["local_depth"],
        "latent_depth": geometry["latent_depth"],
        "lambda_start": float(geometry["lambda_start"]),
        "lambda_local": float(geometry["lambda_local"]),
        "lambda_ramp_start_epoch": 0,
        "lambda_ramp_epochs": 50,
        "local_block_window": geometry["local_block_window"],
        "dtem_window_size": geometry["dtem_window_size"],
        "dtem_feat_dim": 64,
        "dtem_r": 2,
        "dtem_t": 1,
        "total_merge_latent": 0,
        "dtem_train_grouping": "random_per_sample",
        "dtem_train_grouping_seed": 0,
        "soft_topk": True,
        "soft_topk_aux_weight": 0.05,
        "use_softkmax": True,
        "metric_grad_scale": 0.1,
        "source_trace_mode": "center",
        "swa_size": 256,
        "local_cls_global": True,
        "dtem_eval_grouping": FAST_GROUPING,
        "dtem_eval_grouping_seed": 0,
    }
    for name, expected in locked_args.items():
        observed = checkpoint_arg(checkpoint, name)
        require(observed == expected, f"checkpoint args.{name}={observed!r}, expected {expected!r}")
    # Compression budget lives partly in runtime-only _tome_info and therefore
    # is not restored by load_state_dict. Re-apply the preregistered target
    # lambda after loading EMA weights, exactly as the trainer does after the
    # curriculum reaches its target.
    target_lambda = 2.0 if task.model_id == "mn_l2" else 4.0
    applied_merge_count = model.set_compression_lambda(target_lambda)
    evidence.update(
        checkpoint_epoch=199,
        checkpoint_arch=checkpoint.get("arch"),
        ema_state_key=state_key,
        ema_tensor_count=len(cleaned),
        strict_state_dict_load=True,
        effective_lambda=target_lambda,
        effective_total_merge_local=int(applied_merge_count),
    )


def build_loader(task: Task, data_dir: Path, workers: int, torchvision: Any, timm: Any) -> Any:
    dataset = torchvision.datasets.CIFAR100(root=str(data_dir), train=False, download=False)
    require(len(dataset) == EXPECTED_SAMPLES, f"CIFAR-100 test length={len(dataset)}, expected {EXPECTED_SAMPLES}")
    loader = create_eval_loader(task, dataset, workers, timm)
    require(len(loader.dataset) == EXPECTED_SAMPLES, "evaluation loader dataset is not full CIFAR-100 test")
    require(getattr(loader, "drop_last", None) is False, "evaluation loader must not drop the last batch")
    sampler_name = type(getattr(loader, "sampler", None)).__name__
    require(sampler_name == "SequentialSampler", f"evaluation loader sampler is not sequential: {sampler_name}")
    return loader


def create_eval_loader(task: Task, dataset: Any, workers: int, timm: Any) -> Any:
    return timm.data.create_loader(
        dataset,
        input_size=(3, task.resize, task.resize),
        batch_size=task.validation_batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=workers,
        distributed=False,
        crop_pct=0.9,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def gate_from_counts(generic_correct: int, fast_correct: int, sample_count: int) -> Dict[str, Any]:
    require(sample_count == EXPECTED_SAMPLES, f"gate requires exactly {EXPECTED_SAMPLES} samples")
    correct_delta = fast_correct - generic_correct
    delta_pp = correct_delta * 100.0 / sample_count
    passed = abs(correct_delta) <= TOP1_MISMATCH_LIMIT and abs(delta_pp) <= TOP1_DELTA_LIMIT_PP + 1e-12
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "condition": "abs(top1_delta_pp) <= 0.05 and abs(fast_correct-generic_correct) <= 5/10000",
        "threshold_pp": TOP1_DELTA_LIMIT_PP,
        "max_correct_count_difference": TOP1_MISMATCH_LIMIT,
        "correct_count_difference": correct_delta,
    }


def evaluate(
    task: Task,
    model: Any,
    loader: Any,
    torch: Any,
    log_interval: int,
) -> Dict[str, Any]:
    model.cuda().eval()
    generic_correct = 0
    fast_correct = 0
    agreement_count = 0
    sample_count = 0
    abs_diff_sum = 0.0
    abs_diff_count = 0
    max_abs_diff = 0.0
    generic_logit_sum = 0.0
    fast_logit_sum = 0.0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            model.set_dtem_eval_grouping(GENERIC_GROUPING, seed=0)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                generic = extract_logits(model(images)).detach().float()
            model.set_dtem_eval_grouping(FAST_GROUPING, seed=0)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                fast = extract_logits(model(images)).detach().float()
            require(generic.shape == fast.shape, "generic/fast logits shape mismatch")
            require(generic.ndim == 2 and generic.shape[1] == NUM_CLASSES, "unexpected logits shape")
            require(generic.shape[0] == targets.shape[0], "logits/target batch mismatch")
            generic_pred = generic.argmax(dim=1)
            fast_pred = fast.argmax(dim=1)
            difference = (generic - fast).abs()
            batch_size = int(targets.numel())
            sample_count += batch_size
            generic_correct += int((generic_pred == targets).sum().item())
            fast_correct += int((fast_pred == targets).sum().item())
            agreement_count += int((generic_pred == fast_pred).sum().item())
            abs_diff_sum += float(difference.double().sum().item())
            abs_diff_count += int(difference.numel())
            max_abs_diff = max(max_abs_diff, float(difference.max().item()))
            generic_logit_sum += float(generic.double().sum().item())
            fast_logit_sum += float(fast.double().sum().item())
            if batch_index == 0 or (batch_index + 1) % log_interval == 0 or batch_index + 1 == len(loader):
                print(
                    f"[parity] {task.task_id} batch={batch_index + 1}/{len(loader)} "
                    f"samples={sample_count}/{EXPECTED_SAMPLES}",
                    flush=True,
                )
    torch.cuda.synchronize()
    require(sample_count == EXPECTED_SAMPLES, f"evaluated {sample_count}, expected {EXPECTED_SAMPLES}")
    require(abs_diff_count == EXPECTED_SAMPLES * NUM_CLASSES, "logit element count mismatch")
    generic_top1 = generic_correct * 100.0 / sample_count
    fast_top1 = fast_correct * 100.0 / sample_count
    top1_delta = fast_top1 - generic_top1
    mismatch_count = sample_count - agreement_count
    gate = gate_from_counts(generic_correct, fast_correct, sample_count)
    return {
        "dataset": "CIFAR100",
        "split": "test",
        "sample_count": sample_count,
        "class_count": NUM_CLASSES,
        "loader_deterministic": True,
        "loader_shared_between_modes": True,
        "validation_batch_size": task.validation_batch_size,
        "amp": True,
        "amp_dtype": "float16",
        "generic_grouping": GENERIC_GROUPING,
        "fast_grouping": FAST_GROUPING,
        "grouping_seed": 0,
        "generic_correct": generic_correct,
        "fast_correct": fast_correct,
        "generic_top1": generic_top1,
        "fast_top1": fast_top1,
        "top1_delta_pp": top1_delta,
        "abs_top1_delta_pp": abs(top1_delta),
        "argmax_agreement_count": agreement_count,
        "argmax_mismatch_count": mismatch_count,
        "argmax_agreement": agreement_count / sample_count,
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": abs_diff_sum / abs_diff_count,
        "generic_logit_sum": generic_logit_sum,
        "fast_logit_sum": fast_logit_sum,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
        "gate": gate,
    }


def report_identity(
    task: Task,
    campaign: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "runner_revision": RUNNER_REVISION,
        "runner_sha256": hash_file(SCRIPT_PATH),
        "task": task.as_dict(),
        "protocol_file_sha256": campaign["protocol_file_sha256"],
        "protocol_canonical_sha256": campaign["protocol_canonical_sha256"],
        "runtime_tree_sha256": campaign["runtime_tree_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
    }


def reusable_complete_report(
    path: Path,
    identity: Mapping[str, Any],
) -> Tuple[bool, Optional[Mapping[str, Any]]]:
    if path.is_symlink():
        raise ValueError(f"existing report is a symlink: {path}")
    if not path.is_file():
        return False, None
    value = load_json(path)
    require(isinstance(value, Mapping), f"existing report is not an object: {path}")
    existing_identity = value.get("identity")
    require(isinstance(existing_identity, Mapping), f"existing report lacks identity: {path}")
    require(dict(existing_identity) == dict(identity), f"existing report identity drift: {path}; use a new out-dir")
    if value.get("status") != "complete":
        return False, value
    gate = value.get("gate")
    require(isinstance(gate, Mapping) and gate.get("status") in ("PASS", "FAIL"), "complete report has invalid gate")
    return True, value


def execute_task(
    task: Task,
    output: Path,
    protocol: Mapping[str, Any],
    campaign: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    environment: Mapping[str, Any],
    runtime: Tuple[Any, Any, Any],
    args: argparse.Namespace,
) -> str:
    torch, torchvision, timm = runtime
    checkpoint = checkpoint_evidence_stdlib(args.campaign_root, task, campaign)
    identity = report_identity(task, campaign, checkpoint)
    with TaskLock(output):
        reusable, previous = reusable_complete_report(output, identity)
        if reusable:
            status = str(previous["gate"]["status"])
            print(f"[parity] reuse {task.task_id}: {status}", flush=True)
            return status
        attempts = []
        if isinstance(previous, Mapping) and isinstance(previous.get("attempts"), list):
            attempts = list(previous["attempts"])
        attempt = {
            "attempt": len(attempts) + 1,
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
            "gpu_uuid": environment["gpu"]["uuid"],
            "error": None,
        }
        attempts.append(attempt)
        try:
            torch.manual_seed(task.seed)
            torch.cuda.manual_seed_all(task.seed)
            kwargs = model_kwargs(task, protocol)
            model = timm.create_model("mergenet_small_cls", **kwargs)
            load_ema_checkpoint(task, checkpoint, model, torch)
            loader = build_loader(task, args.data_dir, args.workers, torchvision, timm)
            metrics = evaluate(task, model, loader, torch, args.log_interval)
            metrics["fast_vs_training_summary_delta_pp"] = (
                metrics["fast_top1"] - checkpoint["summary_epoch199_fast_ema_top1"]
            )
            metrics["training_summary_comparison_is_diagnostic_only"] = True
            gate = metrics.pop("gate")
            attempt.update(finished_at=utc_now(), status="complete")
            report = {
                "schema_version": SCHEMA_VERSION,
                "runner_revision": RUNNER_REVISION,
                "status": "complete",
                "created_at": previous.get("created_at") if isinstance(previous, Mapping) else attempt["started_at"],
                "updated_at": attempt["finished_at"],
                "identity": identity,
                "task": task.as_dict(),
                "protocol": {
                    "path": str(args.protocol),
                    "file_sha256": campaign["protocol_file_sha256"],
                    "canonical_sha256": campaign["protocol_canonical_sha256"],
                    "post_training_gate": protocol["post_training_release_gate"],
                },
                "runtime": {
                    "root": str(args.runtime_root),
                    "tree_sha256": campaign["runtime_tree_sha256"],
                    "snapshot_manifest_path": campaign["snapshot_manifest_path"],
                    "snapshot_manifest_sha256": campaign["snapshot_manifest_sha256"],
                    "snapshot_bundle_sha256": campaign["snapshot_bundle_sha256"],
                },
                "environment": environment,
                "data": dict(data_identity),
                "checkpoint": checkpoint,
                "model": {
                    "factory": "timm.create_model(mergenet_small_cls)",
                    "kwargs": kwargs,
                    "weights": "epoch_199_state_dict_ema",
                },
                "evaluation": metrics,
                "gate": gate,
                "attempts": attempts,
            }
            atomic_write_json(output, report)
            print(
                f"[parity] complete {task.task_id}: generic={metrics['generic_top1']:.4f} "
                f"fast={metrics['fast_top1']:.4f} delta={metrics['top1_delta_pp']:+.4f}pp "
                f"gate={gate['status']}",
                flush=True,
            )
            return str(gate["status"])
        except Exception as exc:
            attempt.update(
                finished_at=utc_now(),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            error_report = {
                "schema_version": SCHEMA_VERSION,
                "runner_revision": RUNNER_REVISION,
                "status": "error",
                "created_at": previous.get("created_at") if isinstance(previous, Mapping) else attempt["started_at"],
                "updated_at": attempt["finished_at"],
                "identity": identity,
                "task": task.as_dict(),
                "protocol": {
                    "path": str(args.protocol),
                    "file_sha256": campaign["protocol_file_sha256"],
                    "canonical_sha256": campaign["protocol_canonical_sha256"],
                },
                "runtime": {
                    "root": str(args.runtime_root),
                    "tree_sha256": campaign["runtime_tree_sha256"],
                },
                "environment": environment,
                "data": dict(data_identity),
                "checkpoint": checkpoint,
                "gate": {"status": "INCOMPLETE", "pass": False},
                "attempts": attempts,
            }
            atomic_write_json(output, error_report)
            raise
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            try:
                del loader
            except UnboundLocalError:
                pass
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass


def dry_run_plan(
    tasks: Sequence[Task],
    args: argparse.Namespace,
    campaign: Mapping[str, Any],
    data_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = []
    for task in tasks:
        output = args.out_dir / task.filename
        try:
            checkpoint = checkpoint_evidence_stdlib(args.campaign_root, task, campaign)
            readiness = "ready"
            checkpoint_sha = checkpoint["checkpoint_sha256"]
        except Exception as exc:
            readiness = f"not_ready:{type(exc).__name__}:{exc}"
            checkpoint_sha = None
        rows.append(
            {
                **task.as_dict(),
                "checkpoint": str(canonical_job_dir(args.campaign_root, task) / "last.pth.tar"),
                "checkpoint_sha256": checkpoint_sha,
                "output": str(output),
                "output_exists": output.is_file(),
                "readiness": readiness,
            }
        )
    return {
        "dry_run": True,
        "writes_performed": False,
        "torch_imported": "torch" in sys.modules,
        "cuda_initialized": False,
        "campaign_root": str(args.campaign_root),
        "runtime_root": str(args.runtime_root),
        "protocol": str(args.protocol),
        "protocol_file_sha256": campaign["protocol_file_sha256"],
        "protocol_canonical_sha256": campaign["protocol_canonical_sha256"],
        "runtime_tree_sha256": campaign["runtime_tree_sha256"],
        "data": dict(data_identity),
        "full_matrix_task_count": len(MODEL_IDS) * len(RESIZES) * len(SEEDS),
        "selected_task_count": len(tasks),
        "ready_task_count": sum(row["readiness"] == "ready" for row in rows),
        "tasks": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.campaign_root = args.campaign_root.expanduser().resolve(strict=True)
    args.runtime_root = (
        args.runtime_root.expanduser().resolve(strict=True)
        if args.runtime_root
        else (args.campaign_root / "runtime" / "imagenet_longtrain_v1").resolve(strict=True)
    )
    args.protocol = (
        args.protocol.expanduser().resolve(strict=True)
        if args.protocol
        else (args.campaign_root / "runtime" / "cifar_resize_20260810" / "protocol.json").resolve(strict=True)
    )
    args.deps_root = args.deps_root.expanduser().resolve(strict=True)
    args.data_dir = args.data_dir.expanduser().resolve(strict=True)
    args.out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else args.campaign_root / "post_training_parity"
    )
    require(
        args.out_dir == args.campaign_root / "post_training_parity",
        "formal parity output must be CAMPAIGN_ROOT/post_training_parity",
    )
    protocol = load_json(args.protocol)
    require(isinstance(protocol, Mapping), "protocol root must be an object")
    validate_protocol(protocol)
    require(args.workers == 8, "formal parity loader workers must equal protocol.training.workers=8")
    all_tasks = build_tasks(protocol)
    require(len(all_tasks) == 30, "formal parity matrix must contain exactly 30 tasks")
    tasks = select_tasks(all_tasks, args)
    campaign = validate_campaign_snapshot(
        args.campaign_root, args.runtime_root, args.protocol, protocol
    )
    data_identity = validate_data(args.data_dir)
    if args.dry_run:
        print(json.dumps(dry_run_plan(tasks, args, campaign, data_identity), ensure_ascii=False, indent=2))
        return 0

    validate_process_environment(args.runtime_root, args.deps_root)
    gpu = map_physical_gpu(args.gpu)
    torch, torchvision, timm, environment = load_exact_runtime(
        args.runtime_root, args.deps_root, gpu
    )
    statuses: List[str] = []
    errors: List[Dict[str, str]] = []
    for task in tasks:
        output = args.out_dir / task.filename
        try:
            statuses.append(
                execute_task(
                    task,
                    output,
                    protocol,
                    campaign,
                    data_identity,
                    environment,
                    (torch, torchvision, timm),
                    args,
                )
            )
        except Exception as exc:
            errors.append({"task_id": task.task_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[parity][ERROR] {task.task_id}: {errors[-1]['error']}", file=sys.stderr, flush=True)
    summary = {
        "selected": len(tasks),
        "pass": statuses.count("PASS"),
        "fail": statuses.count("FAIL"),
        "errors": errors,
        "out_dir": str(args.out_dir),
        "gpu_uuid": gpu["uuid"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if "FAIL" in statuses:
        return 3
    if errors or len(statuses) != len(tasks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
