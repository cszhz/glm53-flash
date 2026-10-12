#!/usr/bin/env bash
# GLM-5.3-Flash on torch-tpu + vLLM，单台 tpu7x（4 chip / 8 device）TP=8。
#
#   bash /zzlfs/glm53flash_torchtpu/vllm_serve.sh                    # 起 OpenAI server
#   MAX_MODEL_LEN=32768 PORT=8100 bash vllm_serve.sh
#
# 只在空闲机器上跑。sgl-jax 的 dp4 服务占着 .172 和 .152，别动那两台。
#
# 已知边界（2026-09-11）：
#   * 11 层 deepseek_sparse_attention 走 DSA 的 k-pool indexer（jnp 版，还不是
#     Pallas）。GLM53_DENSE_DSA=1 可以退回 dense NoPE MLA —— 两者在
#     <= index_topk=2048 的上下文里数值等价，是 A/B 对拍用的开关。
#   * indexer 打分是 O(num_seqs × tokens × max_pools)，max_pools =
#     MAX_MODEL_LEN/4，中间量 [seqs, max_pools, 4, 256] fp32 会爆内存。现在只能
#     跑小 MAX_MODEL_LEN + 小 MAX_NUM_SEQS，性能等 Pallas kernel。
#   * sparse MLA kernel 只吃 fp8 的 kv_c_normed，所以 KV_CACHE_DTYPE 默认 fp8；
#     对拍 dense 的时候也要带上，否则差的不只是 DSA 这一条路径。
#   * MTP（checkpoint 第 45 层）和 vision tower 的权重在加载时被丢弃，纯文本。
set -Eeuo pipefail

PROJ=/zzlfs/glm53flash_torchtpu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/vllm_env.sh"

MODEL="${MODEL:-/gcs/models/GLM-5.3-Flash}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# 同为 KDA+MLA hybrid 的 Kimi K3 实测：默认 16 的 block 让 100K+ prefill 慢 3~8.5x、
# decode 随上下文变慢；这里 8K 上测得 decode +13%（96 -> 109 tok/s）。
BLOCK_SIZE="${BLOCK_SIZE:-256}"
# decode 一步的 token 数会被 pad 到下一个编译尺寸；max-num-seqs=4 时 decode 只用 1/4。
# prefill 的余数 pad 到 128..8192。桶越少编译越快（每个桶都要编一整张图）。
COMPILE_SIZES="${COMPILE_SIZES:-1,4,8,128,512,1024,2048,4096,8192}"
# 每步 token 预算平分给正在 prefill 的请求：并发长 prompt 一起 prefill、一起开始 decode，
# 而不是串行 prefill 时已经在 decode 的请求每个 token 都陪跑一整个 8K chunk。
# 200K c4 TPOT 111 -> 6.2 ms，c1 不变。
SCHEDULER_CLS="${SCHEDULER_CLS:-vllm_torchtpu.core.decode_aware_chunk_scheduler.SharedPrefillBudgetScheduler}"
# MOE_EP=1：MoE 走 EP8 + fused EP kernel（不再把 token 复制 8 份做 TP GMM）。200K prefill 为主时
# 真实 tok/s/chip +4~8%、TTFT -7%，但 decode 慢（单流 140 -> 98 tok/s，TPOT +1.5~2.8 ms）。
# GSM8K 600：TP 94.83% / EP 95.50%（误差内）。默认 TP。
# TP 下 MLA latent cache 每卡一份，原来每卡都给全部 token gather top-k 行（还每卡各算一遍 indexer）。
# >= 这么多 token 的步改成按 token 切：indexer 只给本卡的 T/8 打分，sparse MLA 两次 all-to-all
# 换成 [T/8, 64 头] 再换回。200K c1 TTFT 16.1 -> 9.7 s，decode 不变。0 = 关。
export TPU_SPARSE_MLA_TOKEN_SHARD_MIN_TOKENS="${TPU_SPARSE_MLA_TOKEN_SHARD_MIN_TOKENS:-128}"
# 同样 >= 128 token 的步走序列并行（glm5_next/sp.py）：残差流按 token 切，mHC/norm 只算 T/8，
# all-reduce 换成 reduce-scatter + all-gather。200K c1 TTFT 9.7 -> 8.3 s。0 = 关。
export GLM53_SP_MIN_TOKENS="${GLM53_SP_MIN_TOKENS:-128}"
# SP 下 11 个 DSA 层用 replicated 的 q_b/o_proj（每卡多 ~1 GB）给本卡 token 算全部 64 头：免掉 q 的
# 两次 all-to-all 和 MLA 层的 reduce-scatter；decode 也走同一路径（全 token 全头）。
# 200K c1 TTFT 8.3 -> 7.4 s，但 decode 139 -> 128 tok/s。decode 为主的负载设 0。
export GLM53_SP_ALL_HEADS="${GLM53_SP_ALL_HEADS:-1}"
EXTRA_MOE_ARGS=()
if [[ "${MOE_EP:-0}" == "1" ]]; then
  export USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_V2_SHARDED_PLAN=1 TPU_MOE_SKIP_PADDED_TOKENS=1
  EXTRA_MOE_ARGS=(--enable-expert-parallel)
fi
# 权重 306 GiB / 8 device = 38 GiB，单卡 HBM 94.74 GiB，给 KV 留 40 GiB。
KV_BYTES="${KV_BYTES:-8589934592}"

# TP=1 起不来：GDN v3 的 KDA kernel 在 64 个 head 不分片时超 VMEM，见
# models/vllm/glm5_next/attention.py 里的注释。
if (( TP < 2 )); then
  echo "TP=$TP 不行：KDA kernel 需要把 64 个 head 分片才装得进 VMEM，用 TP>=2（推荐 8）" >&2
  exit 1
fi

# --kv-cache-memory-bytes 不是调优项，是**必须**给的：不给的话
# TpuWorker.determine_available_memory() 会走 profile_run，而 profile 用的是
# 符号 batch 维，MLA 的空 kv_cache 探测分支里那句 torch.ones(out_shape) 会让
# torch-tpu 的 exporter 报 "SymIntArrayRef expected to contain only concrete
# integers"。给了这个值就直接跳过 profile_run。
echo "=== host $(hostname -I | awk '{print $1}') | $MODEL | TP=$TP | len=$MAX_MODEL_LEN | port=$PORT"

# 权重在 gcsfuse 上，单流只有 ~40 MB/s，306 GiB 要两小时。先并行灌进 page cache
# （944 GiB 内存装得下），之后加载只要 ~70 s，而且同一台机器上后续每次重启都是
# 内存命中。
if [[ "$MODEL" == /gcs/* && "${SKIP_PREFETCH:-0}" != "1" ]]; then
  DIR="$MODEL" bash "$HERE/gcs_prefetch.sh"
fi

echo "=== 预热 venv page cache（NFS）"
vllm_warm_venv

cd /

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name GLM-5.3-Flash \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --attention-backend CUSTOM \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --block-size "$BLOCK_SIZE" \
  --async-scheduling \
  --scheduler-cls "$SCHEDULER_CLS" \
  "${EXTRA_MOE_ARGS[@]}" \
  --compilation-config "{\"compile_sizes\":[$COMPILE_SIZES]}" \
  "$@"
