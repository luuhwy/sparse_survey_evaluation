#!/bin/bash
# SpMM benchmark for RoDe and ASpT on H100 GPU cluster
# Usage: bash run_h100_gpu.sh [output_log]
# Must be run from the repository root directory.

set -euo pipefail

# ── Module loading ────────────────────────────────────────────────────────────
# Ensure the module command is available (source module init if needed)
if ! command -v module &>/dev/null; then
    # Common paths for module system initialization
    for _init in /etc/profile.d/modules.sh \
                 /usr/share/Modules/init/bash \
                 /opt/modules/init/bash; do
        [ -f "$_init" ] && source "$_init" && break
    done
fi

module load CUDA/12.4
module load cmake/3.30.0-rc4

# ── Paths ─────────────────────────────────────────────────────────────────────
# CUDA_HOME is set by 'module load CUDA/12.4'
export CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Repo root (script lives in the repo root)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# ── GPU runner prefix ─────────────────────────────────────────────────────────
GPU_RUN="yhrun -G 1 -p h100x"

# ── Benchmark configuration ───────────────────────────────────────────────────
export GPU_KERNEL=1
export SYSTEM='NVIDIA-H100'

MATRICES=(
    "/root/rootdata/mtx/web-Google/web-Google.mtx"
    "/root/rootdata/mtx/roadNet-CA/roadNet-CA.mtx"
    "/root/rootdata/mtx/cage15/cage15.mtx"
)

K_VALUES=(16 32 64 128 256 512 1024)

# Optional output log file (default: stdout)
LOG="${1:-/dev/stdout}"

# ── Run ───────────────────────────────────────────────────────────────────────
{
echo "=== H100 SpMM Benchmark: RoDe and ASpT ==="
echo "System : ${SYSTEM}"
echo "K values: ${K_VALUES[*]}"
echo ""

for mtx in "${MATRICES[@]}"; do
    echo "===== Matrix: ${mtx} ====="
    for k in "${K_VALUES[@]}"; do
        echo "-------- k=${k}"
        ${GPU_RUN} ./spmm_aspt_gpu.exe "${mtx}" "${k}"
        ${GPU_RUN} ./spmm_rode.exe     "${mtx}" "${k}"
    done
done

echo ""
echo "=== Done ==="
} | tee "${LOG}"
