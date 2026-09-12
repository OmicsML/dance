#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DCCA_DATA_ROOT:-/mnt/nfs/zyxing/data}"
min_free_mb="${DCCA_MIN_FREE_GPU_MB:-18000}"
poll_seconds="${DCCA_GPU_POLL_SECONDS:-60}"
run_count="${DCCA_RUN_COUNT:-300}"
stamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$script_dir/no_cell_qc_logs/$stamp"
cache_dir="$script_dir/no_cell_qc_cache"
mkdir -p "$log_dir"
mkdir -p "$cache_dir/numba" "$cache_dir/matplotlib"
export NUMBA_DISABLE_CACHE=1
export NUMBA_CACHE_DIR="$cache_dir/numba"
export MPLCONFIGDIR="$cache_dir/matplotlib"

find_available_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, -v minimum="$min_free_mb" '{gsub(/ /, "", $2); if ($2 + 0 >= minimum) {print $1; exit}}'
}

gpu_index="${DCCA_GPU_INDEX:-}"
while [[ -z "$gpu_index" ]]; do
    gpu_index="$(find_available_gpu)"
    if [[ -z "$gpu_index" ]]; then
        printf '[%s] Waiting for a GPU with at least %s MiB free.\n' "$(date '+%F %T')" "$min_free_mb"
        sleep "$poll_seconds"
    fi
done

printf '[%s] Using cuda:%s; logs: %s\n' "$(date '+%F %T')" "$gpu_index" "$log_dir"
for task in GSE140203_BRAIN_atac2gex GSE140203_SKIN_atac2gex openproblems_2022_multi_atac2gex; do
    printf '[%s] Starting %s.\n' "$(date '+%F %T')" "$task"
    python "$script_dir/main.py" \
        --root_path "$script_dir" \
        --data_root "$data_root" \
        --subtask "$task" \
        --device "cuda:$gpu_index" \
        --tune_mode pipeline_params \
        --count "$run_count" \
        >"$log_dir/$task.log" 2>&1
    printf '[%s] Finished %s.\n' "$(date '+%F %T')" "$task"
done
