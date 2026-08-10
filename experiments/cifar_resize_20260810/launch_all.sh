#!/usr/bin/env bash
# Freeze the release + harness, validate the locked environment, then launch the
# resumable eight-GPU campaign in a detached process session.
set -Eeuo pipefail

die() { printf '[launch][FATAL] %s\n' "$*" >&2; exit 2; }
info() { printf '[launch] %s\n' "$*"; }

MODE=formal
if [[ "$#" -gt 1 ]]; then
  die "usage: $0 [--dry-run]"
elif [[ "$#" -eq 1 ]]; then
  [[ "$1" == "--dry-run" ]] || die "unknown argument: $1"
  MODE=dry-run
fi

# DRY_RUN belongs only to the per-job audit wrapper. Inheriting it into the
# master can silently turn every 200-epoch job into a no-op, so reject even 0.
[[ ! -v DRY_RUN ]] || die "unset DRY_RUN; use launch_all.sh --dry-run for a launch audit"
for forbidden_name in DEBUG_SUBSET EPOCHS LR MIN_LR MIN_LR_RATIO WARMUP_EPOCHS \
    GLOBAL_BATCH BATCH_SIZE UPDATE_FREQ INITIAL_CHECKPOINT DISTILL_WEIGHT PYTHONOPTIMIZE; do
  [[ ! -v "${forbidden_name}" ]] || die \
    "refusing protocol override environment variable ${forbidden_name}"
done
for distributed_name in WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
    MASTER_ADDR MASTER_PORT; do
  [[ ! -v "${distributed_name}" ]] || die \
    "unset inherited distributed variable ${distributed_name} before launching this single-GPU queue"
done
while IFS='=' read -r environment_name _; do
  [[ "${environment_name}" != TORCHELASTIC_* ]] || die \
    "unset inherited ${environment_name} before launching this single-GPU queue"
done < <(env)
for perturbation_name in CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF \
    PYTORCH_NO_CUDA_MEMORY_CACHING CUBLAS_WORKSPACE_CONFIG CUDA_DEVICE_MAX_CONNECTIONS \
    NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; do
  [[ ! -v "${perturbation_name}" ]] || die \
    "unset inherited CUDA/PyTorch perturbation ${perturbation_name} before launch"
done
for locked_pair in OPENTOME_MERGENET_IMPL=new TIMM_FUSED_ATTN=1; do
  locked_name=${locked_pair%%=*}
  locked_value=${locked_pair#*=}
  [[ ! -v "${locked_name}" || "${!locked_name}" == "${locked_value}" ]] || die \
    "${locked_name} must equal ${locked_value}"
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SOURCE_RUNTIME="${REPO_ROOT}/deliverables/imagenet_longtrain_v1"
SOURCE_HARNESS="${SCRIPT_DIR}"
CAMPAIGN_ROOT=$(realpath -m -- "${CAMPAIGN_ROOT:-${REPO_ROOT}/../otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810}")
DEPS_ROOT=$(realpath -e -- "${DEPS_ROOT:-${REPO_ROOT}/../.deps_mergenet_resize20260810}") \
  || die "exact dependency root is missing"
DATA_DIR=$(realpath -e -- "${DATA_DIR:-${REPO_ROOT}/../data}") \
  || die "CIFAR-100 data root is missing"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

[[ -d "${SOURCE_RUNTIME}" ]] || die "source delivery is missing: ${SOURCE_RUNTIME}"
[[ -f "${SOURCE_HARNESS}/campaign.py" ]] || die "campaign.py is missing"
[[ -f "${SOURCE_HARNESS}/benchmark_resize.py" ]] || die "benchmark_resize.py is missing"
[[ -f "${SOURCE_HARNESS}/run_accuracy_job.sh" ]] || die "run_accuracy_job.sh is missing"
[[ -f "${SOURCE_HARNESS}/protocol.json" ]] || die "protocol.json is missing"
[[ -f "${DATA_DIR}/cifar-100-python/train" && -f "${DATA_DIR}/cifar-100-python/test" ]] \
  || die "DATA_DIR must contain cifar-100-python/{train,test}"
[[ -x /usr/bin/python ]] || die "/usr/bin/python is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
command -v flock >/dev/null 2>&1 || die "flock is unavailable"
command -v setsid >/dev/null 2>&1 || die "setsid is unavailable"
command -v nohup >/dev/null 2>&1 || die "nohup is unavailable"

/usr/bin/python -S - "${GPUS}" <<'PY'
import sys
raw = sys.argv[1]
pieces = raw.split(",")
if len(pieces) != 8 or any(not part.isascii() or not part.isdecimal() for part in pieces):
    raise SystemExit(f"GPUS must contain exactly 8 decimal physical ids: {raw!r}")
ids = [int(part) for part in pieces]
if len(set(ids)) != 8:
    raise SystemExit(f"GPUS contains duplicates: {raw!r}")
PY

mapfile -t PRESENT_GPUS < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | sed 's/[[:space:]]//g'
)
IFS=',' read -r -a REQUESTED_GPUS <<< "${GPUS}"
for gpu in "${REQUESTED_GPUS[@]}"; do
  found=0
  for present in "${PRESENT_GPUS[@]}"; do
    [[ "${gpu}" == "${present}" ]] && found=1
  done
  [[ "${found}" -eq 1 ]] || die "requested physical GPU ${gpu} does not exist"
done

RUNTIME_ROOT="${CAMPAIGN_ROOT}/runtime/imagenet_longtrain_v1"
HARNESS_ROOT="${CAMPAIGN_ROOT}/runtime/cifar_resize_20260810"
SNAPSHOT_MANIFEST="${CAMPAIGN_ROOT}/runtime/snapshot_manifest.json"

write_snapshot_manifest() {
  local output="$1"
  local runtime="$2"
  local harness="$3"
  /usr/bin/python -S - "${output}" "${runtime}" "${harness}" \
    "${SOURCE_RUNTIME}" "${SOURCE_HARNESS}" <<'PY'
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

output, runtime, harness, source_runtime, source_harness = map(Path, sys.argv[1:])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def tree(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        result[str(path.relative_to(root))] = digest(path)
    return result

runtime_files = tree(runtime)
harness_files = tree(harness)
canonical = json.dumps(
    {"runtime": runtime_files, "harness": harness_files},
    sort_keys=True,
    separators=(",", ":"),
).encode()
payload = {
    "schema_version": "mergenet.cifar_resize_snapshot.v1",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "source_runtime": str(source_runtime.resolve()),
    "source_harness": str(source_harness.resolve()),
    "runtime_files": runtime_files,
    "harness_files": harness_files,
    "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

verify_snapshot() {
  /usr/bin/python -S - "${SNAPSHOT_MANIFEST}" "${RUNTIME_ROOT}" "${HARNESS_ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest, runtime, harness = map(Path, sys.argv[1:])
if not manifest.is_file() or not runtime.is_dir() or not harness.is_dir():
    raise SystemExit("runtime snapshot is partial; refusing to repair or overwrite it")
expected = json.loads(manifest.read_text(encoding="utf-8"))

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def tree(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        result[str(path.relative_to(root))] = digest(path)
    return result

actual_runtime = tree(runtime)
actual_harness = tree(harness)
if actual_runtime != expected.get("runtime_files"):
    raise SystemExit("immutable delivery runtime failed hash verification")
if actual_harness != expected.get("harness_files"):
    raise SystemExit("immutable campaign harness failed hash verification")
canonical = json.dumps(
    {"runtime": actual_runtime, "harness": actual_harness},
    sort_keys=True,
    separators=(",", ":"),
).encode()
actual_bundle = hashlib.sha256(canonical).hexdigest()
if actual_bundle != expected.get("bundle_sha256"):
    raise SystemExit("snapshot bundle fingerprint mismatch")
print(f"[launch] immutable snapshot OK: {actual_bundle}")
PY
}

if [[ "${MODE}" == "formal" ]]; then
  mkdir -p "${CAMPAIGN_ROOT}"
  exec 8>"${CAMPAIGN_ROOT}/.runtime_snapshot.lock"
  flock -n -x 8 || die "another launcher is creating/verifying the runtime snapshot"
  if [[ ! -e "${CAMPAIGN_ROOT}/runtime" ]]; then
    TMP_RUNTIME=$(mktemp -d "${CAMPAIGN_ROOT}/.runtime.tmp.XXXXXX")
    trap '[[ -n "${TMP_RUNTIME:-}" && -d "${TMP_RUNTIME}" ]] && rm -rf -- "${TMP_RUNTIME}"' EXIT
    mkdir -p "${TMP_RUNTIME}/imagenet_longtrain_v1" "${TMP_RUNTIME}/cifar_resize_20260810"
    tar -C "${SOURCE_RUNTIME}" --exclude='__pycache__' --exclude='*.pyc' -cf - . \
      | tar -C "${TMP_RUNTIME}/imagenet_longtrain_v1" -xf -
    tar -C "${SOURCE_HARNESS}" --exclude='__pycache__' --exclude='*.pyc' -cf - . \
      | tar -C "${TMP_RUNTIME}/cifar_resize_20260810" -xf -
    write_snapshot_manifest \
      "${TMP_RUNTIME}/snapshot_manifest.json" \
      "${TMP_RUNTIME}/imagenet_longtrain_v1" \
      "${TMP_RUNTIME}/cifar_resize_20260810"
    chmod -R a-w "${TMP_RUNTIME}"
    mv -- "${TMP_RUNTIME}" "${CAMPAIGN_ROOT}/runtime"
    TMP_RUNTIME=""
    trap - EXIT
    info "created immutable runtime snapshot: ${CAMPAIGN_ROOT}/runtime"
  fi
  verify_snapshot
  flock -u 8
  exec 8>&-
else
  # Audit current sources without writing CAMPAIGN_ROOT. Formal launch will copy
  # these exact trees once and then make them read-only.
  RUNTIME_ROOT="${SOURCE_RUNTIME}"
  HARNESS_ROOT="${SOURCE_HARNESS}"
  info "dry-run uses source trees read-only; no runtime snapshot will be created"
fi

PROTOCOL_PATH="${HARNESS_ROOT}/protocol.json"
BENCHMARK_SCRIPT="${HARNESS_ROOT}/benchmark_resize.py"
RUN_ACCURACY_SCRIPT="${HARNESS_ROOT}/run_accuracy_job.sh"

# Locked dependency provenance is checked with CUDA hidden, so this preflight
# cannot initialize or allocate on a card owned by somebody else.
env -u CUDA_VISIBLE_DEVICES \
  CUDA_VISIBLE_DEVICES="" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH="${RUNTIME_ROOT}:${DEPS_ROOT}" \
  /usr/bin/python -S - "${DEPS_ROOT}" "${PROTOCOL_PATH}" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys

deps = Path(sys.argv[1]).resolve()
protocol_path = Path(sys.argv[2]).resolve()
import torch
import torchvision
import timm
import flash_attn

import json
expected = json.loads(protocol_path.read_text(encoding="utf-8"))["expected_environment"]
actual = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "flash_attn": md.version("flash-attn"),
}
if actual != expected:
    raise SystemExit(f"locked dependency mismatch: actual={actual}, expected={expected}")
for module in (torch, torchvision, timm, flash_attn):
    if not Path(module.__file__).resolve().is_relative_to(deps):
        raise SystemExit(f"dependency escaped DEPS_ROOT: {module.__name__}={module.__file__}")
print(f"[launch] dependency provenance OK with CUDA hidden: {actual}")
PY

# Do not initialize CUDA from the launcher. The exact release CUDA tests were
# already run before preregistration; formal evidence starts in benchmark.py,
# which locks a physical UUID and applies two consecutive GPU+host idle gates.
info "launcher CUDA allocation skipped; guarded per-card benchmarks own formal CUDA validation"

BASE_ENV=(
  GPUS="${GPUS}"
  CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  RUNTIME_ROOT="${RUNTIME_ROOT}"
  DEPS_ROOT="${DEPS_ROOT}"
  DATA_DIR="${DATA_DIR}"
  PROTOCOL_PATH="${PROTOCOL_PATH}"
  BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT}"
  RUN_ACCURACY_SCRIPT="${RUN_ACCURACY_SCRIPT}"
  PYTHONPATH="${RUNTIME_ROOT}:${DEPS_ROOT}"
  PYTHONNOUSERSITE=1
  PYTHONDONTWRITEBYTECODE=1
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  OPENTOME_SKIP_OPTIONAL_NLP=1
  CUDA_DEVICE_ORDER=PCI_BUS_ID
  OMP_NUM_THREADS=4
  OPENTOME_MERGENET_IMPL=new
  TIMM_FUSED_ATTN=1
)

if [[ "${MODE}" == "dry-run" ]]; then
  info "printing the complete 45-job master plan; no process will be launched"
  env -u CUDA_VISIBLE_DEVICES -u PYTHONOPTIMIZE -u WORLD_SIZE -u RANK -u LOCAL_RANK \
    -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT \
    "${BASE_ENV[@]}" \
    /usr/bin/python -S "${HARNESS_ROOT}/campaign.py" \
    --protocol "${PROTOCOL_PATH}" --campaign-root "${CAMPAIGN_ROOT}" --dry-run
  exit 0
fi

MASTER_PID_JSON="${CAMPAIGN_ROOT}/state/master.pid"
if [[ -f "${MASTER_PID_JSON}" ]]; then
  EXISTING_PID=$(/usr/bin/python -S - "${MASTER_PID_JSON}" <<'PY'
import json, sys
try:
    print(int(json.load(open(sys.argv[1], encoding="utf-8"))["pid"]))
except Exception:
    print("")
PY
)
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    die "campaign master already alive: pid=${EXISTING_PID}"
  fi
fi

mkdir -p "${CAMPAIGN_ROOT}/logs"
MASTER_LOG="${CAMPAIGN_ROOT}/logs/master.log"
info "launching detached snapshot campaign; log=${MASTER_LOG}"
LAUNCH_STARTED_EPOCH=$(date -u +%s)
nohup setsid env -u DRY_RUN -u CUDA_VISIBLE_DEVICES -u PYTHONOPTIMIZE \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK \
  -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT "${BASE_ENV[@]}" \
  /usr/bin/python -S "${HARNESS_ROOT}/campaign.py" \
    --protocol "${PROTOCOL_PATH}" --campaign-root "${CAMPAIGN_ROOT}" \
  >>"${MASTER_LOG}" 2>&1 </dev/null &
MASTER_PID=$!
printf '%s\n' "${MASTER_PID}" > "${CAMPAIGN_ROOT}/master_launcher.pid"
info "campaign launched: pid=${MASTER_PID}"
info "monitor: /usr/bin/python -S ${HARNESS_ROOT}/monitor.py --campaign-root ${CAMPAIGN_ROOT}"

for _ in $(seq 1 30); do
  if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
    printf '[launch][FATAL] campaign master exited during startup; log tail follows\n' >&2
    tail -n 80 "${MASTER_LOG}" >&2 || true
    exit 2
  fi
  if [[ -s "${CAMPAIGN_ROOT}/state/heartbeat.json" ]]; then
    HEARTBEAT_MATCH=$(/usr/bin/python -S - \
      "${CAMPAIGN_ROOT}/state/heartbeat.json" "${MASTER_PID}" \
      "${LAUNCH_STARTED_EPOCH}" <<'PY'
import datetime as dt
import json
import sys

path, expected_pid, launch_epoch = sys.argv[1:]
try:
    value = json.load(open(path, encoding="utf-8"))
    pid_ok = int(value.get("master_pid")) == int(expected_pid)
    timestamp = dt.datetime.fromisoformat(str(value.get("at", "")).replace("Z", "+00:00"))
    timestamp_ok = (
        timestamp.tzinfo is not None
        and timestamp.timestamp() >= int(launch_epoch)
    )
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    pid_ok = False
    timestamp_ok = False
print("yes" if pid_ok and timestamp_ok else "no")
PY
)
    if [[ "${HEARTBEAT_MATCH}" == "yes" ]]; then
      info "startup verified: master is alive and emitted a fresh matching heartbeat"
      exit 0
    fi
  fi
  sleep 1
done
printf '[launch][FATAL] campaign master is alive but no heartbeat appeared within 30 seconds; log tail follows\n' >&2
tail -n 80 "${MASTER_LOG}" >&2 || true
exit 2
