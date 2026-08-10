#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-${ROOT_DIR}/configs/mergenet_lambda4.yaml}"

info() { printf '[preflight] %s\n' "$*"; }
warn() { printf '[preflight][WARN] %s\n' "$*" >&2; }
die() { printf '[preflight][FATAL] %s\n' "$*" >&2; exit 2; }

GPU_VISIBILITY_HELPER="${SCRIPT_DIR}/gpu_visibility.sh"
[[ -r "${GPU_VISIBILITY_HELPER}" ]] || die "GPU visibility helper is missing: ${GPU_VISIBILITY_HELPER}"
# shellcheck source=gpu_visibility.sh
source "${GPU_VISIBILITY_HELPER}"
configure_gpu_visibility || exit $?

if [[ ! -f "${CONFIG}" && -f "${ROOT_DIR}/${CONFIG}" ]]; then
  CONFIG="${ROOT_DIR}/${CONFIG}"
fi

[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -n "${DATA_DIR:-}" ]] || die 'DATA_DIR is required'
[[ -d "${DATA_DIR}" ]] || die "DATA_DIR is not a directory: ${DATA_DIR}"
[[ -d "${DATA_DIR}/train" ]] || die "missing ImageNet train split: ${DATA_DIR}/train"
[[ -d "${DATA_DIR}/val" ]] || die "missing ImageNet val split: ${DATA_DIR}/val"
[[ -f "${ROOT_DIR}/trainer/classification/in1k_trainer.py" ]] || die 'trainer entrypoint is missing'
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python executable not found: ${PYTHON_BIN}"
command -v "${TORCHRUN_BIN:-torchrun}" >/dev/null 2>&1 || die "torchrun executable not found: ${TORCHRUN_BIN:-torchrun}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
mkdir -p "${OUTPUT_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || die "OUTPUT_DIR is not writable: ${OUTPUT_DIR}"

count_classes() {
  find -L "$1" -mindepth 1 -maxdepth 1 -type d -printf '.' | wc -c
}

train_classes=$(count_classes "${DATA_DIR}/train")
val_classes=$(count_classes "${DATA_DIR}/val")
[[ "${train_classes}" -eq 1000 ]] || die "train must contain 1000 class directories; found ${train_classes}"
[[ "${val_classes}" -eq 1000 ]] || die "val must contain 1000 class directories; found ${val_classes}"

class_diff=$(
  comm -3 \
    <(find -L "${DATA_DIR}/train" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort) \
    <(find -L "${DATA_DIR}/val" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort) \
    | head -n 1
)
[[ -z "${class_diff}" ]] || die 'train and val class-directory names do not match'

find -L "${DATA_DIR}/train" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) -print -quit | grep -q . \
  || die 'no readable image found under train/'
find -L "${DATA_DIR}/val" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) -print -quit | grep -q . \
  || die 'no readable image found under val/'

if [[ "${SKIP_IMAGE_COUNT:-0}" == "1" ]]; then
  warn 'exact ImageNet image-count check skipped explicitly'
else
  expected_train_images="${EXPECTED_TRAIN_IMAGES:-1281167}"
  expected_val_images="${EXPECTED_VAL_IMAGES:-50000}"
  [[ "${expected_train_images}" =~ ^[1-9][0-9]*$ ]] || die 'EXPECTED_TRAIN_IMAGES must be a positive integer'
  [[ "${expected_val_images}" =~ ^[1-9][0-9]*$ ]] || die 'EXPECTED_VAL_IMAGES must be a positive integer'
  train_images=$(find -L "${DATA_DIR}/train" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) -printf '.' | wc -c)
  val_images=$(find -L "${DATA_DIR}/val" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) -printf '.' | wc -c)
  [[ "${train_images}" -eq "${expected_train_images}" ]] \
    || die "train image count must be ${expected_train_images}; found ${train_images}"
  [[ "${val_images}" -eq "${expected_val_images}" ]] \
    || die "val image count must be ${expected_val_images}; found ${val_images}"
  info "dataset image counts OK: train=${train_images}, val=${val_images}"
fi
info "dataset layout OK: train=${train_classes} classes, val=${val_classes} classes"

export OPENTOME_SKIP_OPTIONAL_NLP=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" - "${ROOT_DIR}" "${CONFIG}" "${NPROC_PER_NODE:-0}" <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
requested_workers = int(sys.argv[3])

import yaml
import torch
import torchvision
import timm
import flash_attn
from flash_attn import flash_attn_func

expected = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "timm": "0.9.11",
    "flash-attn": "2.7.4.post1",
}
actual = {
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__.split("+")[0],
    "timm": timm.__version__,
    "flash-attn": metadata.version("flash-attn").split("+")[0],
}
bad = {name: (actual[name], wanted) for name, wanted in expected.items() if actual[name] != wanted}
if bad:
    raise SystemExit(f"dependency version mismatch: {bad}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
visible = torch.cuda.device_count()
if visible < 1:
    raise SystemExit("no visible CUDA device")
if requested_workers and requested_workers > visible:
    raise SystemExit(f"NPROC_PER_NODE={requested_workers} exceeds visible GPUs={visible}")
for index in range(visible if not requested_workers else requested_workers):
    major, minor = torch.cuda.get_device_capability(index)
    if major < 8:
        raise SystemExit(
            f"GPU {index} compute capability {major}.{minor} is unsupported by FlashAttention 2; "
            "Ampere or newer is required"
        )

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
if not isinstance(config, dict):
    raise SystemExit("config must be a YAML mapping")

trainer_path = root / "trainer" / "classification" / "in1k_trainer.py"
spec = importlib.util.spec_from_file_location("mergenet_handoff_trainer", trainer_path)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
allowed = {action.dest for action in trainer.parser._actions}
unknown = sorted(set(config) - allowed)
if unknown:
    raise SystemExit(f"unknown trainer config keys: {unknown}")

required = {
    "dataset": "",
    "train_split": "train",
    "val_split": "val",
    "num_classes": 1000,
    "img_size": 224,
    "patch_size": 8,
    "epochs": 300,
}
wrong = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
if wrong:
    raise SystemExit(f"config violates the ImageNet-1K p8/300e protocol: {wrong}")

launcher_owned = {
    "data_dir", "output", "experiment", "resume", "initial_checkpoint",
    "distill_teacher_checkpoint", "branch_a_checkpoint",
}
present_paths = sorted(key for key in launcher_owned if key in config and config[key] not in (None, ""))
if present_paths:
    raise SystemExit(f"paths/run state must be supplied by the launcher, not YAML: {present_paths}")

model = config.get("model", "")
if model not in {"deit_small_patch16_224", "mergenet_small_cls"}:
    raise SystemExit(f"unexpected handoff model: {model!r}")

print(
    "[preflight] runtime OK: "
    f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
    f"timm={timm.__version__}, flash-attn={actual['flash-attn']}, visible_gpus={visible}"
)
print(
    "[preflight] config OK: "
    f"model={model}, lambda={config.get('lambda_local', 'baseline')}, "
    f"depth={config.get('local_depth', '-') }+{config.get('latent_depth', '-')}, "
    f"epochs={config['epochs']}"
)
PY

run_test() {
  local test_path="$1"
  [[ -f "${test_path}" ]] || die "required test is missing: ${test_path}"
  info "running $(basename "${test_path}")"
  "${PYTHON_BIN}" "${test_path}"
}

info 'running test_launcher_guards.sh'
bash "${ROOT_DIR}/tests/test_launcher_guards.sh"
run_test "${ROOT_DIR}/tests/test_imagefolder_loader.py"
run_test "${ROOT_DIR}/tests/test_accumulation_schedule.py"

if [[ "${SKIP_GPU_TESTS:-0}" == "1" ]]; then
  warn 'GPU attention/model tests skipped; this result is NOT sufficient to start a long run'
else
  run_test "${ROOT_DIR}/tests/test_biased_local_attention.py"
  run_test "${ROOT_DIR}/tests/test_model_smoke.py"
fi

info 'all requested preflight checks passed'
