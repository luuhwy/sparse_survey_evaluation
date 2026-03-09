#!/bin/bash
# SpMM benchmark for RoDe and ASpT on H100 GPU cluster
# Usage: bash run_h100_gpu.sh [output_log]
#   output_log: optional file to tee output into (in addition to stdout)
# Must be run from the repository root directory.

set -uo pipefail   # -u: unset vars are errors; no -e so benchmarks continue on failure

# ── Module loading ────────────────────────────────────────────────────────────
if ! command -v module &>/dev/null; then
    for _init in /etc/profile.d/modules.sh \
                 /usr/share/Modules/init/bash \
                 /opt/modules/init/bash; do
        [ -f "$_init" ] && source "$_init" && break
    done
fi

module load CUDA/12.4
module load cmake/3.30.0-rc4

# ── Paths ─────────────────────────────────────────────────────────────────────
export CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# ── GPU runner prefix ─────────────────────────────────────────────────────────
GPU_RUN="yhrun -G 1 -p h100x"

# ── Benchmark configuration ───────────────────────────────────────────────────
export DATASET='MATRIX_MARKET'   # required by spmm_bench.cpp:128 (no NULL check → SIGSEGV if missing)
export GPU_KERNEL=1
export SYSTEM='NVIDIA-H100'

MATRICES=(
    "/HOME/acict_hpjia/acict_hpjia_1/HDD_POOL/mihongli/suitesparse_all/web-Google/web-Google.mtx"
    "/HOME/acict_hpjia/acict_hpjia_1/HDD_POOL/mihongli/suitesparse_all/roadNet-CA/roadNet-CA.mtx"
    "/HOME/acict_hpjia/acict_hpjia_1/HDD_POOL/mihongli/suitesparse_all/cage15/cage15.mtx"
)

K_VALUES=(16 32 64 128 256)

# ── Optional log file (only used as extra tee target, not as sole output) ─────
LOG="${1:-}"

_run() {
    local exe="$1" mtx="$2" k="$3"
    ${GPU_RUN} "${exe}" "${mtx}" "${k}"
    local ret=$?
    [ $ret -ne 0 ] && echo "  [ERROR] ${exe##*/} exited with code ${ret}" >&2
    return 0   # always continue
}

_main() {
    echo "=== H100 SpMM Benchmark: RoDe and ASpT ==="
    echo "System  : ${SYSTEM}"
    echo "K values: ${K_VALUES[*]}"
    echo ""

    for mtx in "${MATRICES[@]}"; do
        echo "===== Matrix: ${mtx} ====="
        for k in "${K_VALUES[@]}"; do
            echo "-------- k=${k}"
            _run ./spmm_aspt_gpu.exe "${mtx}" "${k}"
            _run ./spmm_rode.exe     "${mtx}" "${k}"
        done
    done

    echo ""
    echo "=== Done ==="
}

# ── Run (tee to log file only if a path was given) ────────────────────────────
if [ -n "${LOG}" ]; then
    _main 2>&1 | tee "${LOG}"
else
    _main
fi
