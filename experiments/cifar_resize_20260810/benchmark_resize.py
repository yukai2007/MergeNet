#!/usr/bin/env python3
"""Per-GPU CIFAR resize efficiency benchmark for the delivered MergeNet.

The default invocation runs the complete synthetic-input matrix on one physical
GPU and atomically updates ``--out`` after every item::

    PYTHONPATH="$RUNTIME_ROOT:$DEPS_ROOT" python benchmark_resize.py \
        --gpu 0 --out state/benchmarks/gpu0.json

Every GPU is intentionally an independent replicate of the complete matrix.
Successful items are reused when the same output is resumed; failed or missing
items are attempted again.  The output lock prevents two processes from
writing or benchmarking the same per-GPU report concurrently.

Protocol (defaults):
  * resize: 160, 192, 224, 256, 320; patch size 8; CIFAR-100 head
  * DeiT-S/8: train and inference
  * MergeNet 6+6, lambda=2, local window=16:
      train/random_per_sample, infer/alternating_per_layer,
      infer/alternating_per_layer_fast, generic-fast logits parity
  * MergeNet 4+8, lambda=4, local window=32: the same four items
  * batch 32, FP16 autocast, 20 warmup and 100 measured iterations

Only the Python standard library is imported before ``CUDA_VISIBLE_DEVICES``
is mapped from the requested physical GPU to its immutable NVIDIA UUID.  This
is deliberate: importing torch before setting visibility can silently benchmark
the wrong card.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "mergenet.cifar_resize_efficiency.v1"
BENCHMARK_REVISION = 2
RESIZES: Tuple[int, ...] = (160, 192, 224, 256, 320)
MODEL_IDS: Tuple[str, ...] = ("deit_s8", "mn_l2", "mn_l4")
MN_MODES: Tuple[str, ...] = (
    "train_random_per_sample",
    "infer_generic",
    "infer_fast",
    "logits_parity",
)
DEIT_MODES: Tuple[str, ...] = ("train", "infer")
PATCH_SIZE = 8
NUM_CLASSES = 100
PARITY_ATOL = 2.0e-2
PARITY_RTOL = 2.0e-2
HOST_LOAD1_PER_CPU_MAX = 1.5

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "deliverables" / "imagenet_longtrain_v1"

EXPECTED_ENVIRONMENT: Mapping[str, str] = {
    "python": "3.10",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "timm": "0.9.11",
    "flash_attn": "2.7.4.post1",
}


MODEL_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "deit_s8": {
        "label": "DeiT-S/8",
        "factory": "timm.create_model(deit_small_patch16_224)",
        "patch_size": PATCH_SIZE,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
    },
    "mn_l2": {
        "label": "MergeNet 6+6 lambda=2 w16",
        "factory": "exact-release:timm.create_model(mergenet_small_cls)",
        "patch_size": PATCH_SIZE,
        "local_depth": 6,
        "latent_depth": 6,
        "lambda_local": 2.0,
        "local_block_window": 16,
        "dtem_window_size": 8,
        "dtem_feat_dim": 64,
        "dtem_t": 1,
        "total_merge_latent": 0,
        "use_softkmax": True,
        # Match the directly trainable delivery YAML, not the older microbench
        # which omitted this flag and therefore measured soft_topk=False.
        "soft_topk": True,
        "soft_topk_aux_weight": 0.05,
        "metric_grad_scale": 0.1,
        "source_trace_mode": "center",
        "swa_size": 256,
        "local_cls_global": True,
        "drop_path_rate": 0.1,
    },
    "mn_l4": {
        "label": "MergeNet 4+8 lambda=4 w32",
        "factory": "exact-release:timm.create_model(mergenet_small_cls)",
        "patch_size": PATCH_SIZE,
        "local_depth": 4,
        "latent_depth": 8,
        "lambda_local": 4.0,
        "local_block_window": 32,
        "dtem_window_size": 8,
        "dtem_feat_dim": 64,
        "dtem_t": 1,
        "total_merge_latent": 0,
        "use_softkmax": True,
        "soft_topk": True,
        "soft_topk_aux_weight": 0.05,
        "metric_grad_scale": 0.1,
        "source_trace_mode": "center",
        "swa_size": 256,
        "local_cls_global": True,
        "drop_path_rate": 0.1,
    },
}


@dataclasses.dataclass(frozen=True)
class Task:
    model_id: str
    resize: int
    mode: str

    @property
    def item_id(self) -> str:
        return f"{self.model_id}_r{self.resize}_{self.mode}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "model_id": self.model_id,
            "resize": self.resize,
            "mode": self.mode,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _host_quality_snapshot() -> Dict[str, Any]:
    """Capture host contention without importing a monitoring dependency."""
    load1, load5, load15 = os.getloadavg()
    system_cpu_count = os.cpu_count()
    try:
        affinity_cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpu_count = None
    effective_cpu_count = affinity_cpu_count or system_cpu_count or 1
    load1_per_cpu = float(load1) / float(effective_cpu_count)
    return {
        "captured_at": _utc_now(),
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
        "os_cpu_count": system_cpu_count,
        "process_affinity_cpu_count": affinity_cpu_count,
        "effective_cpu_count": effective_cpu_count,
        "load1_per_effective_cpu": load1_per_cpu,
        "canonical_max_load1_per_cpu": HOST_LOAD1_PER_CPU_MAX,
        "acceptable_for_timing": load1_per_cpu <= HOST_LOAD1_PER_CPU_MAX,
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace ``path`` without exposing a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Persist the directory entry where the filesystem supports fsync on dirs.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class OutputLock:
    """Non-blocking process lock associated with one output report."""

    def __init__(self, output: Path):
        self.path = Path(f"{output}.lock")
        self._handle: Optional[Any] = None

    def __enter__(self) -> "OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "unknown owner"
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"output is already locked: {self.path} ({owner})"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(
            json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "at": _utc_now()})
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _physical_gpu(value: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("GPU must be a non-negative decimal index")
    if value != "0" and value.startswith("0"):
        raise argparse.ArgumentTypeError("GPU index must not contain leading zeroes")
    return int(value)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one physical GPU's complete MergeNet/DeiT resize benchmark."
    )
    parser.add_argument("--gpu", required=True, type=_physical_gpu,
                        help="Physical nvidia-smi GPU index; safely mapped before torch import.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Per-GPU resumable JSON report (for example gpu0.json).")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--warmup", type=_nonnegative_int, default=20)
    parser.add_argument("--steps", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--runtime-root", type=Path,
                        default=Path(os.environ.get("RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT)))
    parser.add_argument(
        "--deps-root", type=Path,
        default=Path(os.environ["DEPS_ROOT"]) if os.environ.get("DEPS_ROOT") else None,
        help="Exact dependency site-packages root; required via CLI or DEPS_ROOT.",
    )
    parser.add_argument("--protocol", type=Path, default=None,
                        help="Optional external protocol JSON to hash and cross-check.")
    parser.add_argument("--only-model", action="append", choices=MODEL_IDS,
                        help="Repeatable smoke/debug filter; default is all models.")
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS,
                        help="Smoke/debug alias accepting one or more model ids.")
    parser.add_argument("--only-resize", action="append", type=int, choices=RESIZES,
                        help="Repeatable smoke/debug filter; default is all resizes.")
    parser.add_argument("--resizes", nargs="+", type=int, choices=RESIZES,
                        help="Smoke/debug alias accepting one or more resize values.")
    parser.add_argument("--only-mode", action="append", choices=DEIT_MODES + MN_MODES,
                        help="Repeatable smoke/debug filter; default is all modes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the plan and print it without importing torch.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first failed item (successful items remain resumable).")
    parser.add_argument(
        "--allow-env-mismatch", action="store_true",
        help=(
            "Smoke-only override for dependency-version mismatches; output is "
            "permanently marked NONCANONICAL and must not be used as formal evidence."
        ),
    )
    return parser.parse_args(argv)


def _build_tasks(args: argparse.Namespace) -> List[Task]:
    if args.only_model and args.models:
        raise ValueError("use either --only-model or --models, not both")
    if args.only_resize and args.resizes:
        raise ValueError("use either --only-resize or --resizes, not both")
    selected_models = set(args.only_model or args.models or MODEL_IDS)
    selected_resizes = set(args.only_resize or args.resizes or RESIZES)
    selected_modes = set(args.only_mode or (DEIT_MODES + MN_MODES))
    tasks: List[Task] = []
    for resize in RESIZES:
        if resize not in selected_resizes:
            continue
        for model_id in MODEL_IDS:
            if model_id not in selected_models:
                continue
            modes = DEIT_MODES if model_id == "deit_s8" else MN_MODES
            for mode in modes:
                if mode in selected_modes:
                    tasks.append(Task(model_id=model_id, resize=resize, mode=mode))
    if not tasks:
        raise ValueError("filters selected no valid benchmark items")
    return tasks


def _is_filtered(args: argparse.Namespace) -> bool:
    return any(
        (args.only_model, args.models, args.only_resize, args.resizes, args.only_mode)
    )


def _normalize_expected_environment(raw: Mapping[str, Any]) -> Dict[str, str]:
    aliases = {
        "python": ("python", "python_version"),
        "torch": ("torch", "torch_version"),
        "torchvision": ("torchvision", "torchvision_version"),
        "timm": ("timm", "timm_version"),
        "flash_attn": ("flash_attn", "flash-attn", "flash_attn_version"),
    }
    normalized: Dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in raw:
                normalized[canonical] = str(raw[name])
                break
    return normalized


def _read_external_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    path = args.protocol
    if path is None:
        candidate = EXPERIMENT_DIR / "protocol.json"
        path = candidate if candidate.is_file() else None
    if path is None:
        if not _is_filtered(args):
            raise FileNotFoundError(
                "formal full-matrix benchmark requires protocol.json with "
                "expected_environment; filtered smoke runs may omit it"
            )
        return {
            "path": None,
            "sha256": None,
            "validation": "not_present_filtered_smoke",
            "expected_environment": dict(EXPECTED_ENVIRONMENT),
        }
    path = path.expanduser().resolve(strict=True)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    # Protocol authors may place the efficiency section under any of these
    # descriptive keys. Unknown schemas remain pinned by SHA rather than being
    # guessed. Known static fields are checked against this script's contract.
    section = None
    section_name = None
    if isinstance(payload, dict):
        for name in ("efficiency_benchmark", "benchmark", "efficiency"):
            candidate = payload.get(name)
            if isinstance(candidate, dict):
                section = candidate
                section_name = name
                break
    checks: Dict[str, str] = {}
    if isinstance(payload, dict) and "resizes" in payload:
        raw_resizes = payload["resizes"]
        actual_resizes = tuple(
            int(value.get("size", value.get("resize")))
            if isinstance(value, dict) else int(value)
            for value in raw_resizes
        )
        if actual_resizes != RESIZES:
            raise ValueError(
                f"external protocol top-level resizes {actual_resizes} != {RESIZES}"
            )
        checks["top_level_resizes"] = "match"
    if isinstance(payload, dict) and "models" in payload:
        raw_models = payload["models"]
        actual_models = tuple(
            str(value.get("id", value.get("model_id", value.get("name"))))
            if isinstance(value, dict) else str(value)
            for value in raw_models
        )
        if actual_models != MODEL_IDS:
            raise ValueError(
                f"external protocol top-level models {actual_models} != {MODEL_IDS}"
            )
        checks["top_level_models"] = "match"
    if section is not None:
        if "resizes" in section:
            actual = tuple(int(value) for value in section["resizes"])
            if actual != RESIZES:
                raise ValueError(
                    f"external protocol resizes {actual} != built-in {RESIZES}"
                )
            checks["resizes"] = "match"
        raw_models = section.get("model_ids")
        if raw_models is not None:
            actual_models = tuple(str(value) for value in raw_models)
            if actual_models != MODEL_IDS:
                raise ValueError(
                    f"external protocol model_ids {actual_models} != built-in {MODEL_IDS}"
                )
            checks["model_ids"] = "match"
        # On an unfiltered formal run, runtime settings must agree exactly with
        # the preregistered efficiency protocol. Filters intentionally exempt a
        # mini-smoke from this comparison while still pinning the protocol SHA.
        unfiltered = not _is_filtered(args)
        if unfiltered:
            for key, actual in (
                ("batch_size", args.batch_size),
                ("warmup", args.warmup),
                ("steps", args.steps),
            ):
                if key in section and int(section[key]) != actual:
                    raise ValueError(
                        f"external protocol {section_name}.{key}={section[key]} "
                        f"!= CLI {actual}"
                    )
                if key in section:
                    checks[f"{section_name}.{key}"] = "match"

    raw_expected = None
    if isinstance(payload, dict):
        raw_expected = payload.get("expected_environment")
        if raw_expected is None and section is not None:
            raw_expected = section.get("expected_environment")
    if not isinstance(raw_expected, dict):
        if not _is_filtered(args):
            raise ValueError(
                "formal protocol must define expected_environment with Python, "
                "torch, torchvision, timm, and flash-attn versions"
            )
        normalized_expected = dict(EXPECTED_ENVIRONMENT)
        checks["expected_environment"] = "missing_smoke_uses_builtin"
    else:
        normalized_expected = _normalize_expected_environment(raw_expected)
        missing_environment_keys = sorted(
            set(EXPECTED_ENVIRONMENT) - set(normalized_expected)
        )
        if missing_environment_keys:
            raise ValueError(
                "protocol expected_environment is missing: "
                + ", ".join(missing_environment_keys)
            )
        mismatched_protocol_environment = {
            key: {"protocol": normalized_expected[key], "required": required}
            for key, required in EXPECTED_ENVIRONMENT.items()
            if normalized_expected[key] != required
        }
        if mismatched_protocol_environment:
            raise ValueError(
                "protocol expected_environment does not match the release lock: "
                + json.dumps(mismatched_protocol_environment, sort_keys=True)
            )
        checks["expected_environment"] = "match_release_lock"
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "section": section_name,
        "checks": checks,
        "expected_environment": normalized_expected,
        "validation": "known_fields_checked" if checks else "sha_pinned_unknown_schema",
    }


def _query_physical_gpus() -> Dict[int, Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot enumerate physical GPUs with nvidia-smi: {exc}") from exc
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
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def _map_physical_gpu(index: int) -> Dict[str, Any]:
    gpus = _query_physical_gpus()
    if index not in gpus:
        raise ValueError(
            f"physical GPU {index} does not exist; available indices: {sorted(gpus)}"
        )
    gpu = dict(gpus[index])
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    permitted_values = (None, "", str(index), gpu["uuid"])
    if inherited not in permitted_values:
        raise RuntimeError(
            "refusing to reinterpret --gpu under an incompatible inherited "
            f"CUDA_VISIBLE_DEVICES={inherited!r}; unset it or make it {index!r}"
        )
    # UUID mapping is stable even when CUDA and nvidia-smi use different numeric
    # enumeration orders. Inside the process the selected physical card is cuda:0.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    gpu["inherited_cuda_visible_devices"] = inherited
    gpu["exported_cuda_visible_devices"] = gpu["uuid"]
    gpu["process_cuda_index"] = 0
    return gpu


def _prepare_pythonpath(runtime_root: Path, deps_root: Path) -> Tuple[Path, Path]:
    runtime_root = runtime_root.expanduser().resolve(strict=True)
    deps_root = deps_root.expanduser().resolve(strict=True)
    required = (
        runtime_root / "opentome" / "models" / "mergenet" / "model.py",
        runtime_root / "opentome" / "timm" / "dtem.py",
        runtime_root / "opentome" / "timm" / "bias_local_attn.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("runtime root is incomplete: " + ", ".join(missing))

    desired = [str(runtime_root), str(deps_root)]
    sys.path[:] = desired + [entry for entry in sys.path if entry not in desired]
    inherited = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        desired + [entry for entry in inherited if entry not in desired]
    )
    # Some cluster base images combine a new protobuf runtime with an old ONNX
    # generated schema that torchvision imports transitively through timm.  The
    # pure-Python parser avoids that import-time ABI failure; protobuf is not in
    # the timed model path, so this does not affect benchmark measurements.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ["RUNTIME_ROOT"] = str(runtime_root)
    os.environ["DEPS_ROOT"] = str(deps_root)
    return runtime_root, deps_root


def _module_is_under(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _source_hashes(runtime_root: Path) -> Dict[str, str]:
    relative_paths = (
        "opentome/__init__.py",
        "opentome/models/mergenet/model.py",
        "opentome/timm/dtem.py",
        "opentome/timm/bias_local_attn.py",
        "opentome/utils/thetopk.py",
    )
    return {
        relative: _sha256_file(runtime_root / relative) for relative in relative_paths
    }


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_runtime(
    runtime_root: Path, deps_root: Path, physical_gpu: Mapping[str, Any]
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Import and verify the exact release only after GPU visibility is fixed."""
    import torch
    import timm
    import flash_attn
    import opentome
    import opentome.models.mergenet.model as mergenet_model

    if not _module_is_under(opentome, runtime_root):
        raise RuntimeError(
            f"opentome resolved outside exact release: {getattr(opentome, '__file__', None)}"
        )
    if not _module_is_under(mergenet_model, runtime_root):
        raise RuntimeError(
            "MergeNet model resolved outside exact release: "
            f"{getattr(mergenet_model, '__file__', None)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after physical-GPU mapping")
    visible_count = torch.cuda.device_count()
    if visible_count != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device after UUID mapping, got {visible_count}"
        )
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if properties.name.strip() != str(physical_gpu["name"]).strip():
        raise RuntimeError(
            f"mapped GPU name mismatch: torch={properties.name!r}, "
            f"nvidia-smi={physical_gpu['name']!r}"
        )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    source_hashes = _source_hashes(runtime_root)
    source_fingerprint = _sha256_bytes(_canonical_json(source_hashes))
    environment: Dict[str, Any] = {
        "captured_at": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": _distribution_version("torchvision"),
        "timm_version": getattr(timm, "__version__", _distribution_version("timm")),
        "flash_attn_version": getattr(
            flash_attn, "__version__", _distribution_version("flash-attn")
        ),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "amp_dtype": "float16",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "runtime_root": str(runtime_root),
        "deps_root": str(deps_root),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "protobuf_python_implementation": os.environ.get(
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
        ),
        "opentome_module": str(Path(opentome.__file__).resolve()),
        "mergenet_model_module": str(Path(mergenet_model.__file__).resolve()),
        "timm_module": str(Path(timm.__file__).resolve()),
        "source_hashes": source_hashes,
        "source_fingerprint": source_fingerprint,
        "gpu": dict(physical_gpu),
        "torch_gpu": {
            "process_index": 0,
            "name": properties.name,
            "total_memory_mib": int(properties.total_memory // (1024 ** 2)),
            "compute_capability": [properties.major, properties.minor],
        },
    }
    environment_fingerprint_input = {
        key: environment[key]
        for key in (
            "torch_version",
            "torchvision_version",
            "timm_version",
            "flash_attn_version",
            "cuda_runtime_version",
            "cudnn_version",
            "runtime_root",
            "deps_root",
            "source_fingerprint",
            "gpu",
        )
    }
    environment["fingerprint"] = _sha256_bytes(
        _canonical_json(environment_fingerprint_input)
    )
    return torch, timm, environment


def _validate_runtime_environment(
    environment: Dict[str, Any],
    external_protocol: Mapping[str, Any],
    allow_mismatch: bool,
) -> None:
    expected = dict(
        external_protocol.get("expected_environment") or EXPECTED_ENVIRONMENT
    )
    actual = {
        "python": environment["python_version"],
        "torch": environment["torch_version"],
        "torchvision": environment["torchvision_version"],
        "timm": environment["timm_version"],
        "flash_attn": environment["flash_attn_version"],
    }
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key, required in expected.items():
        observed = actual.get(key)
        matches = (
            str(observed).split(".")[:2] == str(required).split(".")[:2]
            if key == "python"
            else observed == required
        )
        if not matches:
            mismatches[key] = {"expected": required, "actual": observed}

    if mismatches and not allow_mismatch:
        raise RuntimeError(
            "formal benchmark environment mismatch: "
            + json.dumps(mismatches, sort_keys=True)
            + "; install the protocol-locked dependencies (do not use the old "
              ".deps_mergenet_delivery directory, which lacks torch)"
        )
    canonical = not mismatches
    validation = {
        "status": "CANONICAL" if canonical else "NONCANONICAL_ENVIRONMENT",
        "canonical": canonical,
        "override_used": bool(mismatches and allow_mismatch),
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
        "warning": None if canonical else (
            "NONCANONICAL SMOKE ONLY — DO NOT USE AS FORMAL EFFICIENCY EVIDENCE"
        ),
    }
    environment["canonical_environment"] = canonical
    environment["noncanonical"] = not canonical
    environment["environment_validation"] = validation


def _protocol_payload(
    args: argparse.Namespace,
    tasks: Sequence[Task],
    environment: Mapping[str, Any],
    external_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_revision": BENCHMARK_REVISION,
        "benchmark_script_sha256": _sha256_file(SCRIPT_PATH),
        "resizes": list(RESIZES),
        "model_ids": list(MODEL_IDS),
        "model_configs": MODEL_CONFIGS,
        "patch_size": PATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "measure_steps": args.steps,
        "amp": True,
        "amp_dtype": "float16",
        "optimizer": "AdamW(lr=1e-3,weight_decay=0.05)",
        "loss": "cross_entropy_synthetic_labels",
        "input": "synthetic_normal_images",
        "seed": args.seed,
        "allow_env_mismatch": bool(args.allow_env_mismatch),
        "canonical_environment": bool(environment["canonical_environment"]),
        "parity_atol": PARITY_ATOL,
        "parity_rtol": PARITY_RTOL,
        "host_quality_policy": {
            "metric": "load1 / process_affinity_cpu_count",
            "maximum": HOST_LOAD1_PER_CPU_MAX,
            "must_pass_at": ["item_start", "item_end"],
            "invalidates": "timing_only",
            "logits_parity_exempt": True,
        },
        "tasks": [task.as_dict() for task in tasks],
        "environment_fingerprint": environment["fingerprint"],
        "external_protocol": dict(external_protocol),
    }


def _compact_item_environment(environment: Mapping[str, Any]) -> Dict[str, Any]:
    gpu = environment["gpu"]
    return {
        "fingerprint": environment["fingerprint"],
        "hostname": environment["hostname"],
        "physical_gpu": gpu["physical_index"],
        "gpu_uuid": gpu["uuid"],
        "gpu_name": gpu["name"],
        "driver_version": gpu["driver_version"],
        "torch_version": environment["torch_version"],
        "timm_version": environment["timm_version"],
        "flash_attn_version": environment["flash_attn_version"],
        "cuda_runtime_version": environment["cuda_runtime_version"],
        "runtime_root": environment["runtime_root"],
        "source_fingerprint": environment["source_fingerprint"],
        "canonical": environment["canonical_environment"],
        "noncanonical": environment["noncanonical"],
        "validation_status": environment["environment_validation"]["status"],
        "warning": environment["environment_validation"]["warning"],
    }


def _autocast(torch: Any) -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _make_grad_scaler(torch: Any) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _extract_logits(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "output", "pred"):
            if key in output:
                return output[key]
        raise TypeError(f"model returned an unsupported dict with keys {sorted(output)}")
    return output


def _model_kwargs(task: Task) -> Dict[str, Any]:
    config = MODEL_CONFIGS[task.model_id]
    if task.model_id == "deit_s8":
        return {
            "pretrained": False,
            "img_size": task.resize,
            "patch_size": PATCH_SIZE,
            "num_classes": NUM_CLASSES,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": config["drop_path_rate"],
        }
    eval_grouping = (
        "alternating_per_layer_fast"
        if task.mode == "infer_fast"
        else "alternating_per_layer"
    )
    return {
        "pretrained": False,
        "num_classes": NUM_CLASSES,
        "img_size": task.resize,
        "patch_size": PATCH_SIZE,
        "local_depth": config["local_depth"],
        "latent_depth": config["latent_depth"],
        "lambda_local": config["lambda_local"],
        "total_merge_latent": config["total_merge_latent"],
        "dtem_window_size": config["dtem_window_size"],
        "dtem_feat_dim": config["dtem_feat_dim"],
        "dtem_t": config["dtem_t"],
        "dtem_train_grouping": "random_per_sample",
        "dtem_train_grouping_seed": 0,
        "dtem_eval_grouping": eval_grouping,
        "dtem_eval_grouping_seed": 0,
        "use_softkmax": config["use_softkmax"],
        "metric_grad_scale": config["metric_grad_scale"],
        "source_trace_mode": config["source_trace_mode"],
        "swa_size": config["swa_size"],
        "local_block_window": config["local_block_window"],
        "local_cls_global": config["local_cls_global"],
        "soft_topk": config["soft_topk"],
        "soft_topk_aux_weight": config["soft_topk_aux_weight"],
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": config["drop_path_rate"],
    }


def _build_model(task: Task, torch: Any, timm: Any, seed: int) -> Any:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = _model_kwargs(task)
    if task.model_id == "deit_s8":
        model = timm.create_model("deit_small_patch16_224", **kwargs)
    else:
        # The exact release module was imported and path-verified in
        # _load_runtime; create_model now resolves that registered factory.
        model = timm.create_model("mergenet_small_cls", **kwargs)
    return model.cuda()


def _token_metadata(task: Task) -> Dict[str, Any]:
    patches = (task.resize // PATCH_SIZE) ** 2
    result: Dict[str, Any] = {
        "input_patch_tokens": patches,
        "input_tokens_with_cls": patches + 1,
    }
    if task.model_id.startswith("mn_"):
        compression = float(MODEL_CONFIGS[task.model_id]["lambda_local"])
        total_merge = int(patches * (compression - 1.0) / compression)
        result["retained_patch_tokens"] = patches - total_merge
        result["nominal_token_retention"] = (patches - total_merge) / patches
    return result


def _base_item(
    task: Task,
    args: argparse.Namespace,
    environment: Mapping[str, Any],
    protocol_fingerprint: str,
    attempt: int,
) -> Dict[str, Any]:
    return {
        **task.as_dict(),
        "schema_version": SCHEMA_VERSION,
        "benchmark_revision": BENCHMARK_REVISION,
        "protocol_fingerprint": protocol_fingerprint,
        "attempt": attempt,
        "started_at": _utc_now(),
        "finished_at": None,
        "success": False,
        "oom": False,
        "error": None,
        "traceback": None,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "measure_steps": args.steps,
        "model_init_seed": args.seed,
        "synthetic_input_seed": args.seed + task.resize,
        "amp": True,
        "amp_dtype": "float16",
        "model_config": dict(MODEL_CONFIGS[task.model_id]),
        "model_factory": MODEL_CONFIGS[task.model_id]["factory"],
        "effective_model_kwargs": _model_kwargs(task),
        "tokens": _token_metadata(task),
        "params": None,
        "parameter_count": None,
        "trainable_params": None,
        "step_time_ms": None,
        "latency_ms_per_batch": None,
        "wall_step_time_ms": None,
        "throughput_img_s": None,
        "train_img_s": None,
        "infer_img_s": None,
        "allocated_after_warmup_mib": None,
        "reserved_after_warmup_mib": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "host_quality_start": None,
        "host_quality_end": None,
        "timing_valid": None,
        "timing_invalid_reason": None,
        "environment": _compact_item_environment(environment),
    }


def _run_timed_item(
    task: Task,
    item: Dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    timm: Any,
) -> None:
    if task.mode == "logits_parity":
        raise ValueError("parity item passed to timed benchmark")
    model = _build_model(task, torch, timm, args.seed)
    item["params"] = int(sum(parameter.numel() for parameter in model.parameters()))
    item["parameter_count"] = item["params"]
    item["trainable_params"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    input_generator = torch.Generator(device="cuda")
    input_generator.manual_seed(args.seed + task.resize)
    images = torch.randn(
        args.batch_size, 3, task.resize, task.resize,
        generator=input_generator, device="cuda", dtype=torch.float32,
    )
    training = task.mode in ("train", "train_random_per_sample")
    if training:
        model.train()
        if task.model_id.startswith("mn_"):
            model.set_dtem_train_grouping("random_per_sample", seed=0)
        labels = torch.randint(
            0, NUM_CLASSES, (args.batch_size,), generator=input_generator,
            device="cuda", dtype=torch.long,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1.0e-3,
            weight_decay=0.05,
        )
        scaler = _make_grad_scaler(torch)
    else:
        model.eval()
        labels = None
        optimizer = None
        scaler = None

    last_logits = None
    last_loss = None

    def one_step() -> None:
        nonlocal last_logits, last_loss
        if training:
            assert optimizer is not None and scaler is not None and labels is not None
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch):
                last_logits = _extract_logits(model(images))
                last_loss = torch.nn.functional.cross_entropy(last_logits, labels)
            scaler.scale(last_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.inference_mode(), _autocast(torch):
                last_logits = _extract_logits(model(images))

    torch.cuda.synchronize()
    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    item["allocated_after_warmup_mib"] = float(
        torch.cuda.memory_allocated() / (1024 ** 2)
    )
    item["reserved_after_warmup_mib"] = float(
        torch.cuda.memory_reserved() / (1024 ** 2)
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # The host-quality boundary must describe the measured window, not model
    # construction or warmup.  Capture the actual load immediately before the
    # first timed CUDA event so only contention relevant to timing is judged.
    item["host_quality_start"] = _host_quality_snapshot()
    wall_started = time.perf_counter()
    start_event.record()
    for _ in range(args.steps):
        one_step()
    end_event.record()
    end_event.synchronize()
    wall_elapsed = time.perf_counter() - wall_started
    item["host_quality_end"] = _host_quality_snapshot()
    cuda_elapsed_ms = float(start_event.elapsed_time(end_event))

    item["step_time_ms"] = cuda_elapsed_ms / args.steps
    item["latency_ms_per_batch"] = item["step_time_ms"]
    item["wall_step_time_ms"] = wall_elapsed * 1000.0 / args.steps
    item["throughput_img_s"] = args.batch_size * args.steps / (cuda_elapsed_ms / 1000.0)
    item["peak_allocated_mib"] = float(
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )
    item["peak_reserved_mib"] = float(
        torch.cuda.max_memory_reserved() / (1024 ** 2)
    )
    if training:
        item["train_img_s"] = item["throughput_img_s"]
        item["last_loss"] = float(last_loss.detach().float().item())
        item["grad_scaler_scale"] = float(scaler.get_scale())
    else:
        item["infer_img_s"] = item["throughput_img_s"]
    # Materialize a scalar outside the timed region, proving output production.
    item["output_checksum"] = float(last_logits.detach().float().sum().item())
    item["success"] = True


def _run_parity_item(
    task: Task,
    item: Dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    timm: Any,
) -> None:
    if not task.model_id.startswith("mn_") or task.mode != "logits_parity":
        raise ValueError("invalid parity task")
    model = _build_model(task, torch, timm, args.seed)
    model.eval()
    item["params"] = int(sum(parameter.numel() for parameter in model.parameters()))
    item["parameter_count"] = item["params"]
    item["trainable_params"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + task.resize)
    images = torch.randn(
        args.batch_size, 3, task.resize, task.resize,
        generator=generator, device="cuda", dtype=torch.float32,
    )
    with torch.inference_mode():
        model.set_dtem_eval_grouping("alternating_per_layer", seed=0)
        with _autocast(torch):
            generic = _extract_logits(model(images)).detach().float()
        torch.cuda.synchronize()

        model.set_dtem_eval_grouping("alternating_per_layer_fast", seed=0)
        with _autocast(torch):
            fast = _extract_logits(model(images)).detach().float()
        torch.cuda.synchronize()

    difference = (generic - fast).abs()
    denominator = generic.abs().clamp_min(1.0e-8)
    allclose = bool(torch.allclose(generic, fast, atol=PARITY_ATOL, rtol=PARITY_RTOL))
    cosine = torch.nn.functional.cosine_similarity(generic, fast, dim=1).mean()
    argmax_agreement = (generic.argmax(dim=1) == fast.argmax(dim=1)).float().mean()
    item.update(
        parity_generic_grouping="alternating_per_layer",
        parity_fast_grouping="alternating_per_layer_fast",
        parity_atol=PARITY_ATOL,
        parity_rtol=PARITY_RTOL,
        allclose=allclose,
        max_abs_diff=float(difference.max().item()),
        mean_abs_diff=float(difference.mean().item()),
        rms_diff=float(difference.square().mean().sqrt().item()),
        max_relative_diff=float((difference / denominator).max().item()),
        cosine_similarity=float(cosine.item()),
        argmax_agreement=float(argmax_agreement.item()),
        generic_checksum=float(generic.sum().item()),
        fast_checksum=float(fast.sum().item()),
        peak_allocated_mib=float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        peak_reserved_mib=float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
    )
    item["success"] = allclose
    if not allclose:
        item["error"] = (
            "generic-fast logits parity failed: "
            f"max_abs={item['max_abs_diff']:.6g}, atol={PARITY_ATOL}, rtol={PARITY_RTOL}"
        )


def _cleanup_cuda(torch: Any) -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def _execute_item(
    task: Task,
    args: argparse.Namespace,
    torch: Any,
    timm: Any,
    environment: Mapping[str, Any],
    protocol_fingerprint: str,
    attempt: int,
) -> Dict[str, Any]:
    item = _base_item(
        task, args, environment, protocol_fingerprint=protocol_fingerprint, attempt=attempt
    )
    # Timed items overwrite these fields immediately around their CUDA-event
    # measurement.  The fallback snapshots retain diagnostics for a failure
    # during model construction/warmup; logits parity is timing-exempt.
    item["host_quality_start"] = None
    item["host_quality_exempt"] = task.mode == "logits_parity"
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if task.mode == "logits_parity":
            _run_parity_item(task, item, args, torch, timm)
        else:
            _run_timed_item(task, item, args, torch, timm)
    except torch.cuda.OutOfMemoryError as exc:
        item["oom"] = True
        item["error"] = f"CUDA OOM: {str(exc).splitlines()[0]}"
        item["traceback"] = traceback.format_exc()
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            item["oom"] = True
        item["error"] = f"RuntimeError: {message.splitlines()[0]}"
        item["traceback"] = traceback.format_exc()
    except Exception as exc:  # keep the remaining matrix resumable
        item["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        item["traceback"] = traceback.format_exc()
    finally:
        if item.get("host_quality_start") is None:
            item["host_quality_start"] = _host_quality_snapshot()
        if item.get("host_quality_end") is None:
            item["host_quality_end"] = _host_quality_snapshot()
        if task.mode == "logits_parity":
            # Parity is a numerical statement, not a timing measurement.
            item["timing_valid"] = None
            item["timing_invalid_reason"] = None
        elif not item.get("success"):
            item["timing_valid"] = False
            item["timing_invalid_reason"] = "benchmark_item_failed"
        else:
            invalid_points = [
                point
                for point in ("start", "end")
                if not item[f"host_quality_{point}"]["acceptable_for_timing"]
            ]
            item["timing_valid"] = not invalid_points
            if invalid_points:
                item["timing_invalid_reason"] = (
                    "host_overloaded_at_" + "_and_".join(invalid_points)
                    + f": load1/effective_cpu > {HOST_LOAD1_PER_CPU_MAX}"
                )
        item["finished_at"] = _utc_now()
        _cleanup_cuda(torch)
    return item


def _new_report(
    args: argparse.Namespace,
    physical_gpu: Mapping[str, Any],
    environment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_revision": BENCHMARK_REVISION,
        "protocol_fingerprint": protocol_fingerprint,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "complete": False,
        "all_success": False,
        "physical_gpu": physical_gpu["physical_index"],
        "gpu_uuid": physical_gpu["uuid"],
        "output": str(args.out.expanduser().resolve()),
        "protocol": dict(protocol),
        "environment": dict(environment),
        "canonical_environment": environment["canonical_environment"],
        "noncanonical": environment["noncanonical"],
        "environment_warning": environment["environment_validation"]["warning"],
        "current_item": None,
        "items": [],
        "summary": {},
    }


def _load_or_create_report(
    args: argparse.Namespace,
    physical_gpu: Mapping[str, Any],
    environment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
) -> Dict[str, Any]:
    output = args.out.expanduser().resolve()
    if not output.exists():
        return _new_report(
            args, physical_gpu, environment, protocol, protocol_fingerprint
        )
    with output.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"cannot resume {output}: schema {report.get('schema_version')!r} "
            f"!= {SCHEMA_VERSION!r}"
        )
    if report.get("protocol_fingerprint") != protocol_fingerprint:
        raise RuntimeError(
            f"cannot resume {output}: protocol/environment fingerprint changed; "
            "use a new output path to keep measurements scientifically separate"
        )
    if report.get("physical_gpu") != physical_gpu["physical_index"]:
        raise RuntimeError(
            f"cannot resume {output}: it belongs to physical GPU "
            f"{report.get('physical_gpu')}, not {physical_gpu['physical_index']}"
        )
    if report.get("gpu_uuid") != physical_gpu["uuid"]:
        raise RuntimeError(
            f"cannot resume {output}: GPU UUID changed from {report.get('gpu_uuid')} "
            f"to {physical_gpu['uuid']}"
        )
    if not isinstance(report.get("items"), list):
        raise RuntimeError(f"cannot resume {output}: items must be a list")
    report["environment"] = dict(environment)
    report["canonical_environment"] = environment["canonical_environment"]
    report["noncanonical"] = environment["noncanonical"]
    report["environment_warning"] = environment["environment_validation"]["warning"]
    report["protocol"] = dict(protocol)
    report["updated_at"] = _utc_now()
    report["complete"] = False
    report["all_success"] = False
    report["current_item"] = None
    return report


def _summarize(report: Mapping[str, Any], tasks: Sequence[Task]) -> Dict[str, Any]:
    expected = {task.item_id for task in tasks}
    latest = {
        item.get("item_id"): item
        for item in report.get("items", [])
        if item.get("item_id") in expected
    }
    successful = sum(bool(item.get("success")) for item in latest.values())
    failed = sum(not bool(item.get("success")) for item in latest.values())
    missing = len(expected - set(latest))
    oom = sum(bool(item.get("oom")) for item in latest.values())
    timed_success_items = [
        item for item in latest.values()
        if item.get("mode") != "logits_parity" and item.get("success") is True
    ]
    timing_valid = sum(
        item.get("timing_valid") is True for item in timed_success_items
    )
    timing_invalid = sum(
        item.get("timing_valid") is not True for item in timed_success_items
    )
    parity_success = sum(
        item.get("mode") == "logits_parity" and item.get("success") is True
        for item in latest.values()
    )
    return {
        "expected_items": len(expected),
        "recorded_items": len(latest),
        "successful_items": successful,
        "failed_items": failed,
        "oom_items": oom,
        "missing_items": missing,
        "timing_valid_items": timing_valid,
        "timing_invalid_items": timing_invalid,
        "parity_successful_items": parity_success,
        "all_recorded_timings_valid": timing_invalid == 0,
    }


def _replace_item(report: Dict[str, Any], replacement: Dict[str, Any]) -> None:
    item_id = replacement["item_id"]
    items = [item for item in report["items"] if item.get("item_id") != item_id]
    items.append(replacement)
    order = {
        task.item_id: index
        for index, task in enumerate(
            Task(model_id, resize, mode)
            for resize in RESIZES
            for model_id in MODEL_IDS
            for mode in (DEIT_MODES if model_id == "deit_s8" else MN_MODES)
        )
    }
    items.sort(key=lambda item: order.get(item.get("item_id"), len(order)))
    report["items"] = items


def _print_item(item: Mapping[str, Any]) -> None:
    compact = {
        key: item.get(key)
        for key in (
            "item_id",
            "success",
            "oom",
            "error",
            "throughput_img_s",
            "step_time_ms",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "timing_valid",
            "timing_invalid_reason",
            "max_abs_diff",
            "allclose",
        )
        if key in item
    }
    print("BENCH_ITEM " + json.dumps(compact, ensure_ascii=False), flush=True)


def _run(args: argparse.Namespace) -> int:
    tasks = _build_tasks(args)
    if args.allow_env_mismatch and not _is_filtered(args):
        raise ValueError(
            "--allow-env-mismatch is smoke-only and requires --models, --resizes, "
            "or another explicit task filter"
        )
    external_protocol = _read_external_protocol(args)
    if args.dry_run:
        payload = {
            "gpu": args.gpu,
            "out": str(args.out),
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "steps": args.steps,
            "runtime_root": str(args.runtime_root),
            "deps_root": None if args.deps_root is None else str(args.deps_root),
            "allow_env_mismatch": args.allow_env_mismatch,
            "external_protocol": external_protocol,
            "tasks": [task.as_dict() for task in tasks],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.deps_root is None:
        raise ValueError(
            "DEPS_ROOT/--deps-root is required and must point to the exact locked "
            "environment; there is intentionally no fallback to the old "
            ".deps_mergenet_delivery directory"
        )

    output = args.out.expanduser().resolve()
    with OutputLock(output):
        physical_gpu = _map_physical_gpu(args.gpu)
        runtime_root, deps_root = _prepare_pythonpath(args.runtime_root, args.deps_root)
        torch, timm, environment = _load_runtime(
            runtime_root, deps_root, physical_gpu
        )
        _validate_runtime_environment(
            environment,
            external_protocol=external_protocol,
            allow_mismatch=args.allow_env_mismatch,
        )
        protocol = _protocol_payload(
            args, tasks, environment=environment, external_protocol=external_protocol
        )
        protocol_fingerprint = _sha256_bytes(_canonical_json(protocol))
        report = _load_or_create_report(
            args,
            physical_gpu,
            environment,
            protocol,
            protocol_fingerprint,
        )
        existing = {item.get("item_id"): item for item in report["items"]}

        for task in tasks:
            previous = existing.get(task.item_id)
            previous_is_reusable = (
                previous is not None
                and previous.get("success") is True
                and (
                    task.mode == "logits_parity"
                    or previous.get("timing_valid") is True
                )
            )
            if previous_is_reusable:
                print(f"BENCH_SKIP {task.item_id} already successful", flush=True)
                continue
            attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
            report["current_item"] = task.item_id
            report["updated_at"] = _utc_now()
            report["summary"] = _summarize(report, tasks)
            _atomic_write_json(output, report)
            print(
                f"BENCH_START {task.item_id} gpu={args.gpu} "
                f"batch={args.batch_size} warmup={args.warmup} steps={args.steps}",
                flush=True,
            )
            item = _execute_item(
                task,
                args,
                torch,
                timm,
                environment,
                protocol_fingerprint,
                attempt,
            )
            _replace_item(report, item)
            existing[task.item_id] = item
            report["current_item"] = None
            report["updated_at"] = _utc_now()
            report["summary"] = _summarize(report, tasks)
            _atomic_write_json(output, report)
            _print_item(item)
            if not item.get("success") and args.fail_fast:
                break

        summary = _summarize(report, tasks)
        report["summary"] = summary
        report["current_item"] = None
        report["updated_at"] = _utc_now()
        report["complete"] = (
            summary["missing_items"] == 0
            and summary["failed_items"] == 0
            and summary["timing_invalid_items"] == 0
        )
        report["all_success"] = (
            report["complete"] and summary["successful_items"] == summary["expected_items"]
        )
        _atomic_write_json(output, report)
        print(
            "BENCH_REPORT "
            + json.dumps(
                {
                    "output": str(output),
                    "complete": report["complete"],
                    "all_success": report["all_success"],
                    **summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if report["all_success"] else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("benchmark interrupted; the last completed item remains resumable", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"BENCH_FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
