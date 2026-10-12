#!/usr/bin/env bash
# torch-tpu + vLLM 这条链路的公共环境。被 vllm_smoke.sh / 后续的 serve 脚本 source。
# 单机模式：把本机从「4 主机 slice 的成员」降级成「独立的 1 主机 slice」。
#
# 用法：
#   source /zzlfs/glm53flash_torchtpu/vllm_env.sh          # 单机（默认）
#   NHOSTS=2 source /zzlfs/glm53flash_torchtpu/vllm_env.sh # 预留：多机走 ray，见下
#
# 骨架取自 /zzlfs/qwen397/start_dp_server.sh，那些注释是实打实踩出来的，别删。

PROJ=/zzlfs/glm53flash_torchtpu
VENV="${VENV:-$PROJ/.venv}"
SRC="${SRC:-$PROJ/vllm-torchtpu}"

# --- 单机 TPU 拓扑覆盖 -------------------------------------------------------
# 实例元数据（mig-0905 的每一台）说自己是 2x2x4 slice 的成员：
#   TPU_ACCELERATOR_TYPE=tpu7x-32  TOPOLOGY=2x2x4
#   CHIPS_PER_HOST_BOUNDS=2,2,1（每台 4 chip）  HOST_BOUNDS=1,1,4（4 台）
# 不覆盖就直接起，libtpu 会去等另外 3 台的 8471，单机永远等不到。
#   HOST_BOUNDS 1,1,4 -> 1,1,1
#   TOPOLOGY = chips_per_host × host_bounds 逐位相乘 = 2x2x1
#   ACCELERATOR_TYPE 的数字是 **device 数**：4 chip × 2 core = 8 -> tpu7x-8
export TPU_PROCESS_ADDRESSES=localhost:8471
export TPU_WORKER_HOSTNAMES=localhost
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_TOPOLOGY=2x2x1
export TPU_WORKER_ID=0
unset TPU_MULTIHOST_BACKEND
# 让 libtpu 别回头查元数据，查到的是那套 4 机的值，会把上面的覆盖顶掉。
export TPU_SKIP_MDS_QUERY=1

# --- 插件 ---------------------------------------------------------------------
if [[ -d "$SRC/src/vllm_torchtpu" ]]; then
  export PYTHONPATH="$SRC/src${PYTHONPATH:+:$PYTHONPATH}"
fi
export PJRT_DEVICE=TPU
export VLLM_TARGET_DEVICE=tpu
# 两个 entry point 都要列出来：`torchtpu` 是 platform plugin，`torchtpu_layers`
# 是 general plugin，后者才是调用 register_models() 的那个。VLLM_PLUGINS 是一份
# 白名单，只写 `torchtpu` 的话模型注册会被静默跳过 —— 不会报错，vLLM 会把
# Glm5NextForConditionalGeneration 解析成 Transformers fallback。
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export SKIP_JAX_PRECOMPILE=1
# sparse MLA 的 SparseCore cache 布局（main 的默认）要求 chunk_size*topk 被
# num_owners*occurrences（2048）整除。GLM-5.3 的 topk 是 2048 pool token + 3 个尾部
# token = 2051（buffer 补到 2052），过不了。先用 9 月验证过的 TensorCore 布局；
# SparseCore 要把 topk 补到 128 的倍数（2176），是后面的优化项。
export TPU_SPARSE_MLA_NOPE_LAYOUT="${TPU_SPARSE_MLA_NOPE_LAYOUT:-tensorcore}"
export TPU_SPARSE_MLA_ROPE_LAYOUT="${TPU_SPARSE_MLA_ROPE_LAYOUT:-tensorcore}"

# TorchTPU 的 split compiler 产物不可序列化，vLLM 编译缓存和 AOTAutograd 缓存都得关。
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
# inductor 产物默认落根盘，冷启动多 worker 能写几十 G，写爆之后全体报 ENOSPC。
# 挪到 /dev/shm，比根盘快，重启自清。
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/torchinductor}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
# README 里点名的 TorchTPU 临时 workaround。
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS="${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-false}"

export VLLM_XLA_CHECK_RECOMPILATION=0
export RAY_memory_monitor_refresh_ms=0
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# venv 在 NFS 上，冷启动 `import vllm` 要 60 s；vLLM CLI 会用 spawn 重开子进程再 import
# 一遍，而 DP coordinator 只给 30 s（coordinator.py 写死），冷启动必然超时。
# 先并行读一遍 .so/.py 把 NFS page cache 捂热 —— 只读文件、不 import、不碰 TPU。
vllm_warm_venv() {
  find "$VENV/lib" -type f \( -name '*.so' -o -name '*.py' \) -print0 2>/dev/null \
    | xargs -0 -r -P 16 cat > /dev/null 2>&1 || true
}
