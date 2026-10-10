#!/usr/bin/env bash
set -euo pipefail
program_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pipeline_python="${PIPELINE_PYTHON:-/mnt/pfs/Data/wanghao/Data_Clean_StridingAI_Ego/.venv_root_miniforge/bin/python}"
if [[ ! -x "$pipeline_python" ]]; then
  printf 'Python interpreter missing: %s. Set PIPELINE_PYTHON.\n' "$pipeline_python" >&2
  exit 1
fi
exec "$pipeline_python" "$program_dir/run_s5.py" "$@"
