#!/usr/bin/env bash
# Shared GPUS validation for the launcher and the standalone preflight entry.

configure_gpu_visibility() {
  local gpu_spec="${GPUS:-}"
  local gpu_id
  local -a gpu_ids=()
  local -A seen_gpu_ids=()

  [[ -n "${gpu_spec}" ]] || return 0
  if [[ "${gpu_spec}" == *' '* ]]; then
    printf '[gpu-visibility][FATAL] GPUS must be a comma-separated list without spaces\n' >&2
    return 2
  fi
  if [[ ! "${gpu_spec}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf '[gpu-visibility][FATAL] invalid GPUS list: %s\n' "${gpu_spec}" >&2
    return 2
  fi

  IFS=',' read -r -a gpu_ids <<< "${gpu_spec}"
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ "${gpu_id}" != "0" && ! "${gpu_id}" =~ ^[1-9][0-9]*$ ]]; then
      printf '[gpu-visibility][FATAL] GPU ids must not contain leading zeroes: %s\n' "${gpu_id}" >&2
      return 2
    fi
    if [[ -n "${seen_gpu_ids[${gpu_id}]+x}" ]]; then
      printf '[gpu-visibility][FATAL] duplicate GPU id in GPUS: %s\n' "${gpu_id}" >&2
      return 2
    fi
    seen_gpu_ids["${gpu_id}"]=1
  done

  export CUDA_VISIBLE_DEVICES="${gpu_spec}"
}
