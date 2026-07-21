set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python"
PIPELINE_TITLE="default_pipeline"
BASE_TITLE=""
POST_TITLE=""
CACHE_SAVE_DIR="${PROJECT_ROOT}/cache/${PIPELINE_TITLE}_base_cache"
CACHE_DIR_OVERRIDDEN=false

BASE_TRAIN_STEPS=100100
POST_TRAIN_STEPS=100100
BASE_ROUND=""
TASK_NAME="default_task"
TASK_CFG_PATH=""
TASK_OVERRIDE=""
RUN_TAG="$(date +%Y.%m.%d/%H%M%S)"
BASE_GPUS="6"
POST_GPUS="6"
EXPORT_GPU="6"
BASE_PORT=""
POST_PORT=""

usage() {
  cat <<'USAGE_EOF'
Usage:
  bash scripts/pipeline.sh [options]

Options:
  --title <str>             Pipeline title prefix.
                            Default: default_pipeline
                            Base title defaults to "<title>_base"
                            Post title defaults to "<title>_post"

  --base-title <str>        Override stage-1 Save_Title.
  --post-title <str>        Override stage-2 Save_Title.
  --cache-dir <path>        KV cache output directory used by export+post.

  --base-steps <int>        Stage-1 num_train_steps (base).
  --post-steps <int>        Stage-2 num_train_steps (post).
  --base-round <int>        Snapshot step used for export/post (e.g. 90000).
                            If omitted, uses the largest snapshot in base run dir.
  --task-name <str>         Task config name under cfgs/suite/task/<name>.yaml.
                            Example: default_task
                            This is passed to stage-1 as "suite/task=<name>".
                            For export+post, script checks base run task_picked
                            matches this name to keep task launch synchronized.

  --base-gpus <csv>         Stage-1 GPU list, e.g. 1 or 1,2,3. Default: 1
  --post-gpus <csv>         Stage-2 GPU list. Default: same as --base-gpus
  --export-gpu <id>         Export-stage single GPU id. Default: first base GPU

  --base-port <int>         Stage-1 MASTER_PORT (local_host_id). Default: auto free port
  --post-port <int>         Stage-2 MASTER_PORT (local_host_id). Default: auto free port

  --run-tag <str>           Fixed run subpath under title directories.
                            Default: current time "YYYY.MM.DD/HHMMSS"
  --python <bin>            Python executable. Default: python
  --help                    Show this help.

Example:
  bash scripts/pipeline.sh \
    --title experiment_name \
    --task-name default_task \
    --base-gpus 0 \
    --post-gpus 0 \
    --export-gpu 0 \
    --base-port 22351 \
    --post-port 24351 \
    --cache-dir /path/to/cache_dir \
    --base-steps 101000 \
    --post-steps 120000 \
    --base-round 90000
USAGE_EOF
}

normalize_gpu_csv() {
  local csv="${1// /}"
  if [[ ! "${csv}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] Invalid GPU list: ${1}. Expected format like 1 or 1,2,3" >&2
    exit 1
  fi
  echo "${csv}"
}

first_gpu_from_csv() {
  local csv="$1"
  echo "${csv%%,*}"
}

to_hydra_list() {
  local csv="$1"
  echo "[${csv}]"
}

validate_port_number() {
  local p="$1"
  if [[ ! "${p}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid port: ${p}" >&2
    exit 1
  fi
  if (( p < 1024 || p > 65535 )); then
    echo "[ERROR] Port out of range [1024,65535]: ${p}" >&2
    exit 1
  fi
}

validate_task_name() {
  local task_name="$1"
  if [[ ! "${task_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] Invalid --task-name: ${task_name}" >&2
    echo "        Only [A-Za-z0-9._-] are allowed." >&2
    exit 1
  fi
}

pick_two_free_ports() {
  "${PYTHON_BIN}" - <<'PY'
import socket

s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s1.bind(("", 0))
p1 = s1.getsockname()[1]

s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.bind(("", 0))
p2 = s2.getsockname()[1]

print(f"{p1} {p2}")

s1.close()
s2.close()
PY
}

assert_port_free() {
  local p="$1"
  "${PYTHON_BIN}" - "${p}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("", port))
except OSError:
    print(f"[ERROR] Port already in use: {port}", file=sys.stderr)
    sys.exit(1)
finally:
    s.close()
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)
      PIPELINE_TITLE="$2"
      shift 2
      ;;
    --base-title)
      BASE_TITLE="$2"
      shift 2
      ;;
    --post-title)
      POST_TITLE="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_SAVE_DIR="$2"
      CACHE_DIR_OVERRIDDEN=true
      shift 2
      ;;
    --base-steps)
      BASE_TRAIN_STEPS="$2"
      shift 2
      ;;
    --post-steps)
      POST_TRAIN_STEPS="$2"
      shift 2
      ;;
    --base-round)
      BASE_ROUND="$2"
      shift 2
      ;;
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --base-gpus)
      BASE_GPUS="$2"
      shift 2
      ;;
    --post-gpus)
      POST_GPUS="$2"
      shift 2
      ;;
    --export-gpu)
      EXPORT_GPU="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --post-port)
      POST_PORT="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${TASK_NAME}" ]]; then
  validate_task_name "${TASK_NAME}"
  TASK_CFG_PATH="${PROJECT_ROOT}/cfgs/suite/task/${TASK_NAME}.yaml"
  if [[ ! -f "${TASK_CFG_PATH}" ]]; then
    echo "[ERROR] Task config not found: ${TASK_CFG_PATH}" >&2
    echo "        Please use --task-name with an existing cfg name under cfgs/suite/task/" >&2
    exit 1
  fi
  TASK_OVERRIDE="suite/task=${TASK_NAME}"
fi

if [[ "${CACHE_DIR_OVERRIDDEN}" == false ]]; then
  CACHE_SAVE_DIR="${PROJECT_ROOT}/cache/${PIPELINE_TITLE}_base_cache"
fi

if [[ -z "${BASE_TITLE}" ]]; then
  BASE_TITLE="${PIPELINE_TITLE}_base"
fi
if [[ -z "${POST_TITLE}" ]]; then
  POST_TITLE="${PIPELINE_TITLE}_post"
fi

BASE_GPUS="$(normalize_gpu_csv "${BASE_GPUS}")"
if [[ -z "${POST_GPUS}" ]]; then
  POST_GPUS="${BASE_GPUS}"
fi
POST_GPUS="$(normalize_gpu_csv "${POST_GPUS}")"

if [[ -z "${EXPORT_GPU}" ]]; then
  EXPORT_GPU="$(first_gpu_from_csv "${BASE_GPUS}")"
fi
if [[ ! "${EXPORT_GPU}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Invalid --export-gpu: ${EXPORT_GPU}" >&2
  exit 1
fi

BASE_GPU_HYDRA="$(to_hydra_list "${BASE_GPUS}")"
POST_GPU_HYDRA="$(to_hydra_list "${POST_GPUS}")"

AUTO_BASE_PORT=""
AUTO_POST_PORT=""
if [[ -z "${BASE_PORT}" || -z "${POST_PORT}" ]]; then
  read -r AUTO_BASE_PORT AUTO_POST_PORT < <(pick_two_free_ports)
fi
if [[ -z "${BASE_PORT}" ]]; then
  BASE_PORT="${AUTO_BASE_PORT}"
fi
if [[ -z "${POST_PORT}" ]]; then
  POST_PORT="${AUTO_POST_PORT}"
fi

validate_port_number "${BASE_PORT}"
validate_port_number "${POST_PORT}"
if [[ "${BASE_PORT}" == "${POST_PORT}" ]]; then
  echo "[ERROR] base-port and post-port are identical (${BASE_PORT}); please set different ports" >&2
  exit 1
fi
assert_port_free "${BASE_PORT}"
assert_port_free "${POST_PORT}"

BASE_RUN_DIR="${PROJECT_ROOT}/checkpoints/${BASE_TITLE}/${RUN_TAG}"
POST_RUN_DIR="${PROJECT_ROOT}/checkpoints_post/${POST_TITLE}/${RUN_TAG}"

echo "========== Pipeline Config =========="
echo "PROJECT_ROOT      : ${PROJECT_ROOT}"
echo "PYTHON_BIN        : ${PYTHON_BIN}"
echo "PIPELINE_TITLE    : ${PIPELINE_TITLE}"
echo "BASE_TITLE        : ${BASE_TITLE}"
echo "POST_TITLE        : ${POST_TITLE}"
echo "CACHE_SAVE_DIR    : ${CACHE_SAVE_DIR}"
echo "BASE_TRAIN_STEPS  : ${BASE_TRAIN_STEPS}"
echo "POST_TRAIN_STEPS  : ${POST_TRAIN_STEPS}"
echo "TASK_NAME         : ${TASK_NAME:-<hydra default>}"
if [[ -n "${TASK_CFG_PATH}" ]]; then
  echo "TASK_CFG_PATH     : ${TASK_CFG_PATH}"
fi
echo "BASE_GPUS         : ${BASE_GPUS}"
echo "POST_GPUS         : ${POST_GPUS}"
echo "EXPORT_GPU        : ${EXPORT_GPU}"
echo "BASE_PORT         : ${BASE_PORT}"
echo "POST_PORT         : ${POST_PORT}"
echo "RUN_TAG           : ${RUN_TAG}"
echo "BASE_RUN_DIR      : ${BASE_RUN_DIR}"
echo "POST_RUN_DIR      : ${POST_RUN_DIR}"
echo "====================================="

cd "${PROJECT_ROOT}"
mkdir -p "${CACHE_SAVE_DIR}"

echo
echo "[1/3] Stage-1 Base Training"
"${PYTHON_BIN}" scripts/train_ddp_base.py \
  Save_Title="${BASE_TITLE}" \
  num_train_steps="${BASE_TRAIN_STEPS}" \
  "${TASK_OVERRIDE}" \
  multi_gpu="${BASE_GPU_HYDRA}" \
  local_host_id="${BASE_PORT}" \
  root_dir="${PROJECT_ROOT}" \
  hydra.run.dir="${BASE_RUN_DIR}" \
  hydra.sweep.dir="${BASE_RUN_DIR}"

if [[ ! -d "${BASE_RUN_DIR}/snapshot" ]]; then
  echo "[ERROR] Snapshot directory not found: ${BASE_RUN_DIR}/snapshot" >&2
  exit 1
fi

if [[ -z "${BASE_ROUND}" ]]; then
  mapfile -t _rounds < <(
    find "${BASE_RUN_DIR}/snapshot" -maxdepth 1 -type f -name "*.pt" \
      -printf "%f\n" \
      | sed 's/\.pt$//' \
      | grep -E '^[0-9]+$' \
      | sort -n
  )
  if [[ ${#_rounds[@]} -eq 0 ]]; then
    echo "[ERROR] No numeric checkpoints found in ${BASE_RUN_DIR}/snapshot" >&2
    exit 1
  fi
  BASE_ROUND="${_rounds[-1]}"
fi

BASE_CKPT="${BASE_RUN_DIR}/snapshot/${BASE_ROUND}.pt"
BASE_CONFIG="${BASE_RUN_DIR}/.hydra/config.yaml"
BASE_STATS="${BASE_RUN_DIR}/stats.hdf5"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[ERROR] Base checkpoint missing: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] Base config missing: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${BASE_STATS}" ]]; then
  echo "[ERROR] Base stats missing: ${BASE_STATS}" >&2
  exit 1
fi

if [[ -n "${TASK_NAME}" ]]; then
  BASE_TASK_PICKED="$(
    grep -E '^[[:space:]]*task_picked:[[:space:]]*' "${BASE_CONFIG}" \
      | head -n1 \
      | sed -E 's/^[[:space:]]*task_picked:[[:space:]]*//'
  )"
  if [[ -z "${BASE_TASK_PICKED}" ]]; then
    echo "[ERROR] Cannot find task_picked in base config: ${BASE_CONFIG}" >&2
    exit 1
  fi
  if [[ "${BASE_TASK_PICKED}" != "${TASK_NAME}" ]]; then
    echo "[ERROR] Task mismatch detected." >&2
    echo "        --task-name=${TASK_NAME}" >&2
    echo "        base config task_picked=${BASE_TASK_PICKED}" >&2
    echo "        Please use matching --base-title/--run-tag, or change --task-name." >&2
    exit 1
  fi
  echo "Validated task sync: ${BASE_TASK_PICKED}"
fi

echo "Selected base checkpoint: ${BASE_CKPT}"

echo
echo "[2/3] Export Base KV Cache"
CUDA_VISIBLE_DEVICES="${EXPORT_GPU}" "${PYTHON_BIN}" scripts/export_cache/export_base_cache.py \
  --ckpt "${BASE_CKPT}" \
  --save-dir "${CACHE_SAVE_DIR}"

_cache_count="$(find "${CACHE_SAVE_DIR}" -type f -name "cache_*.hdf5" | wc -l | awk '{print $1}')"
if [[ "${_cache_count}" -eq 0 ]]; then
  echo "[ERROR] No cache files found under ${CACHE_SAVE_DIR}" >&2
  exit 1
fi
echo "Exported cache files: ${_cache_count}"

echo
echo "[3/3] Stage-2 Post Training"
"${PYTHON_BIN}" scripts/train_ddp_post.py \
  Save_Title="${POST_TITLE}" \
  num_train_steps="${POST_TRAIN_STEPS}" \
  multi_gpu="${POST_GPU_HYDRA}" \
  local_host_id="${POST_PORT}" \
  cache_path="${CACHE_SAVE_DIR}" \
  base_dir="${BASE_RUN_DIR}" \
  round="${BASE_ROUND}" \
  base_weight_dir="${BASE_CKPT}" \
  base_config_dir="${BASE_CONFIG}" \
  base_stats_path="${BASE_STATS}" \
  root_dir="${PROJECT_ROOT}" \
  hydra.run.dir="${POST_RUN_DIR}" \
  hydra.sweep.dir="${POST_RUN_DIR}"

echo
echo "Pipeline complete."
echo "Base run : ${BASE_RUN_DIR}"
echo "Post run : ${POST_RUN_DIR}"
echo "Cache dir: ${CACHE_SAVE_DIR}"
