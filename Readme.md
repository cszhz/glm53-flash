# GLM-5.3-Flash on vllm-torchtpu

本文档汇总 GLM-5.3-Flash（FP8）在 TPU v7x 上用 vllm-torchtpu 服务的拉起方法、测试脚本和压测结果。
文中引用的脚本、工具和结果文件都在本目录（`/zzlfs/glm53flash_torchtpu/guide`）下：

```
guide/
├── Readme.md
├── scripts/   vllm_install.sh  vllm_env.sh  vllm_serve.sh  restart_serve.sh  router.sh  gcs_prefetch.sh
├── tools/     功能 / 性能测试、profile、对拍脚本（含 ctx200k_bench.py、dp_router.py、prefix_hit_ab.py）
├── patches/   vllm-torchtpu 上的改动：prefix cache（0001–0002，3.6 节）、MTP 投机解码（0003–0005，3.7 节）
└── results/   5.1 节、3.6 节压测和精度的原始数据
```

venv（`.venv`）、代码（`vllm-torchtpu/`）、日志（`logs/`）仍在项目根目录 `/zzlfs/glm53flash_torchtpu`；
下面的命令都在项目根目录执行。

---

## 1. 概览

| 项目     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 模型     | GLM-5.3-Flash FP8（`GLM-5.3-Flash`，`model_type: glm5_next`），纯文本（视觉塔不加载；MTP 层可选加载做投机解码，见 3.7 节） |
| 硬件     | tpu7x-32 slice 中的两台：10.128.0.89（worker 2）、10.128.0.38（worker 3），每台 4 chip / 8 device |
| 部署拓扑 | **DP2 × TP8**：每台一个独立的 TP8 实例，prefill 叠加序列并行（SP）；本机 :8080 路由按最少在途请求分发 |
| 代码     | `vllm-torchtpu/` 的 `glm53-port` 分支，`19f6a3239`（= 上游 PR 的内容 + prefix cache + 调度器修正 + MTP 投机解码，见 3.6、3.7 节；五个 patch 在 `guide/patches/`） |
| Prefix cache | 支持：`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 --enable-prefix-caching`，命中粒度 2048 token，见 3.6 节 |
| 投机解码 | MTP（`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`），需要 unified pool；单流 decode 1.8–2.3×，见 3.7 节 |
| 上游 PR  | [vllm-project/vllm-torchtpu#1859](https://github.com/vllm-project/vllm-torchtpu/pull/1859)（branch `cszhz/glm53-flash`，rebase 到 `72deb3860`；该基线的 `pyproject.toml` 钉的是 libtpu 0.0.50，本指南统一用 0.0.47，换法见第 2 节） |
| 软件     | vLLM 0.29.0（源码构建）、torch-tpu release 2026.9.22、**libtpu 0.0.47**、jax 0.11.2、transformers 5.17.0 |

**200K 上下文核心结果（DP2，8 chip）**：c1 TTFT **7.46 s**、TPOT **6.14 ms**；饱和吞吐 **~6800 tok/s/chip**
（同硬件 sglang-jax 最终版：c1 TTFT 7.41 s、TPOT 12.5 ms、~3700 tok/s/chip）。

---

## 2. 环境安装

```bash
cd /zzlfs/glm53flash_torchtpu
bash guide/scripts/vllm_install.sh   # 建 .venv：vLLM 0.29.0 源码构建 + torch-tpu + libtpu；transformers 单独抬到 5.17.0
```

- venv 在 NFS 上（`/zzlfs/glm53flash_torchtpu/.venv`），两台机器共用，无需各自安装。
- `guide/scripts/vllm_env.sh` 负责公共环境：把本机从 4 机 slice 降级成独立单机 slice（`TPU_HOST_BOUNDS=1,1,1`、
  `TPU_SKIP_MDS_QUERY=1` 等）、插件白名单、sparse MLA 的 TensorCore cache 布局、inductor 缓存目录。
  同目录的 `vllm_serve.sh` 会自动 source 它。
- **libtpu 用 0.0.47**（本文所有结果都在 0.0.47 上测得）。服务用的 `vllm-torchtpu/`（`glm53-port`，基线 `28aa22a9b`）
  本身钉的就是 0.0.47。如果从 PR 分支（`cszhz/glm53-flash`，钉 0.0.50）安装，装完后换回：
  ```bash
  .venv/bin/pip install "libtpu==0.0.47" "tpu-sync-torch==0.0.1.dev20261002213151"
  ```
  （`guide/scripts/vllm_install.sh` 最后一步已经做了这件事。）确认：服务日志里有 `Setting libtpu version to: 0.0.47`。原因见第 7 节第 4 条。

---

## 3. 拉起服务

### 3.1 单机（一个 TP8 实例）

```bash
cd /zzlfs/glm53flash_torchtpu
# 200K 上下文配置（压测用的就是这个）
MAX_MODEL_LEN=204800 KV_BYTES=34359738368 bash guide/scripts/restart_serve.sh <tag>
# 日志：logs/<tag>.log；端口 8000；冷启动约 35 分钟（权重 prefetch + 各 token 桶编译）
curl -s localhost:8000/health          # 200 即就绪
```

- `guide/scripts/restart_serve.sh`：先 SIGINT 停掉旧服务（停不下就杀进程组），清 `/dev/shm/torchinductor`，再后台起同目录的 `vllm_serve.sh`。
  **不要连续快速调用两次**，两个实例会抢芯片，报 `PjRtClient is not initialized`。
- 远程机器：`ssh 10.128.0.38 'MAX_MODEL_LEN=204800 KV_BYTES=34359738368 bash /zzlfs/glm53flash_torchtpu/guide/scripts/restart_serve.sh <tag>'`
- 用其它代码目录（例如开发 worktree）：加 `SRC=/path/to/worktree`。
- 启动过程中**不要修改正在使用的源码**：worker 是分批 import 的，新旧文件混用会导致启动失败。

### 3.2 两机 DP2 + 路由

```bash
# 两台各起一个实例（见 3.1），都健康后：
bash guide/scripts/router.sh http://127.0.0.1:8000 http://10.128.0.38:8000
curl -s localhost:8080/router/stats    # 各后端在途/已处理请求数；routing: 粘性命中/溢出/无键请求数
```

路由（`guide/tools/dp_router.py`）按**前缀粘性**分发：两台各有自己的 prefix cache，同一会话的每一轮都要落到同一台。
粘性键是 chat 请求的 system 消息 + 第一条 user 消息（completions 取 prompt 开头，或请求头 `X-Session-Id`）。
新会话分给在途最少的一台并记住（最近 20 万个会话，LRU），之后同一会话回到记住的那台；记住的那台比最空的一台
多出 `--spill`（默认 4）个在途请求时，溢出到最空的一台并改记那台。没有键的请求（GET 等）选在途最少的。
所以一次性的、prompt 各不相同的请求（比如压测）和原来的"最少在途"分发一样均衡。没有健康检查：一台挂掉，分到它的请求会失败。

对外地址：`http://10.128.0.89:8080`（OpenAI 兼容接口，模型名 `GLM-5.3-Flash`）。

快速自检：

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"爱因斯坦出生在哪里？"}],"max_tokens":1024,"temperature":0}'
```

### 3.3 `guide/scripts/vllm_serve.sh` 主要参数（环境变量覆盖）

| 变量                     | 默认                           | 说明                                                         |
| ------------------------ | ------------------------------ | ------------------------------------------------------------ |
| `MAX_MODEL_LEN`          | 8192                           | 200K 用 204800                                               |
| `KV_BYTES`               | 8 GiB                          | 每 device KV 预算；200K 用 32 GiB（每副本可并发 ~10 条 200K） |
| `MAX_NUM_SEQS`           | 8                              | 每副本最大并发                                               |
| `MAX_NUM_BATCHED_TOKENS` | 8192                           | 每步 token 预算（prefill chunk）                             |
| `BLOCK_SIZE`             | 256                            | KV 页大小                                                    |
| `COMPILE_SIZES`          | `1,4,8,128,...,8192`           | 编译的 token 桶                                              |
| `SCHEDULER_CLS`          | `SharedPrefillBudgetScheduler` | 每步 prefill 预算平分给在 prefill 的请求，避免并发时 TPOT 被长 prefill 拖垮 |
| `KV_CACHE_DTYPE`         | fp8                            | sparse MLA kernel 只支持 fp8                                 |
| `MOE_EP`                 | 0                              | 1 = MoE 走 EP8（prefill +~8%，decode 140 -> 98 tok/s，默认关） |

固定参数：`--no-enable-prefix-caching`、`--async-scheduling`、`--attention-backend CUSTOM`。

### 3.4 性能开关（都已在 `guide/scripts/vllm_serve.sh` 默认打开）

| 变量                                    | 默认 | 作用                                                         | 关掉时                       |
| --------------------------------------- | ---- | ------------------------------------------------------------ | ---------------------------- |
| `TPU_SPARSE_MLA_TOKEN_SHARD_MIN_TOKENS` | 128  | 每步 >= 128 token 时 sparse MLA / indexer 按 token 切给 8 张卡 | 0                            |
| `GLM53_SP_MIN_TOKENS`                   | 128  | prefill 序列并行：残差流按 token 切，mHC/norm 只算 1/8，all-reduce 换成 reduce-scatter + all-gather | 0                            |
| `GLM53_SP_ALL_HEADS`                    | 1    | SP 下 DSA 层用 replicated q_b/o_proj 给本卡 token 算全部 64 头（每卡 +~1 GB）。200K TTFT -11%，但单流 decode 139 -> 128 tok/s | 0（decode 为主的负载建议关） |
| `TPU_MHC_GATES_PALLAS`                  | 1    | mHC gates 用 Pallas kernel                                   | 0                            |
| `GLM53_INDEXER_V1`                      | 0    | 1 = 用 XLA 参考版 indexer（A/B 精度用）                      |                              |
| `GLM53_DENSE_DSA`                       | 0    | 1 = DSA 层退回稠密 MLA（A/B 用，<2048 token 时与稀疏等价）   |                              |

### 3.5 并行方式

- **机器间**：DP2，两台互不通信。
- **机器内**：TP8。KDA（34 层线性注意力）按 head 切；MoE（288 专家）按专家中间维度切（TP，不是 EP）。
- **prefill（>= 128 token）**：叠加 SP。残差流 / mHC / norm 按 token 切；KDA、MoE 输入 all-gather、输出 reduce-scatter；
  11 个 DSA（sparse MLA）层每卡给本卡 token 算全部头，只 all-gather 768 维的 KV/indexer 行；indexer 只给本卡 token 打分。
- **decode**：KDA / MoE 走 TP（每层 all-reduce），DSA 层每卡全 token 全头（无通信）。
- MLA latent KV cache 每卡一份完整的；KDA 递推状态按 head 切。

---

### 3.6 Prefix cache（unified KV pool）

```bash
# 两台都这样起（vllm_serve.sh 里写死的 --no-enable-prefix-caching 会被后面的参数覆盖）
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 MODEL=/home/zzl_google_com/models/GLM-5.3-Flash \
MAX_MODEL_LEN=204800 KV_BYTES=34359738368 bash guide/scripts/restart_serve.sh <tag> \
  --enable-prefix-caching --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
```

**为什么要改代码**：混合模型（34 层 KDA + 11 层 sparse MLA）开 prefix cache，KDA 的递归状态必须存在调度器管理的
block 里（`mamba_cache_mode=align`），TPU 上就是 unified KV pool。原生的 unified pool 给每层同一块 attention 形状的
tensor，GLM-5.3 的 sparse MLA（`(nope, rope)` 两块）和 DSA indexer 的 pool-key cache 都接不上。

**做法**（`guide/patches/0001-*.patch`，主要在 `glm5_next/unified_pool.py`）：每个 pool 区域分配 GLM-5.3 原生的三块 cache，
一个 block 正好一页：

| 区域里的 tensor | 每 token | 谁用 |
| --- | --- | --- |
| nope `(blocks, bs, 4, 128)` uint8 | 512 B | sparse MLA；KDA 状态（fp32 ssm 1024 行 + bf16 conv 64 行）；indexer 的 member 状态（前 4 行） |
| rope `(blocks, bs/4, 4, 128)` uint8 | 128 B | sparse MLA |
| pool key `(blocks, bs/16, 4, 256)` uint8 | 64 B | DSA indexer；并进 MLA 组共用它的 block 表（单独成组会让显存翻倍） |

- sparse MLA / indexer 的 kernel 一行没改。block 大小取能装下一层 KDA 状态的最小 2 的幂：1088 行 → **2048**。
- KDA 走 Kimi 的 pooled op（新增 `pool_tokens_per_row=1`）；member 状态用 `pool_adapters.gather_region/scatter_region` 读写。
  用 XLA gather/scatter 会让 uint8 pool 整块转布局（每层每步 2.3 GB，decode 慢 5 倍），单测
  `guide/tools/test_indexer_pooled_member.py` 会检查编译结果里没有整块 pool 的 copy。
- 不设 `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1` 时走原来的 per-layer cache，行为不变。

**验证**（.184，200K / TP8；`guide/tools/prefix_hit_ab.py`）：

| 测试 | 结果 |
| --- | --- |
| 同一 prompt 发两次（2045~82043 token，含 2048 边界 ±1、±3） | 命中数 = ⌊(n−1)/2048⌋×2048，输出与重算**逐位一致**（logprob 差 0） |
| 共享前缀：P+S2 命中 P+S1 缓存的 P（P 到 123K） | 与清空缓存后重算逐位一致 |
| 16 请求并发 × 2 轮，共享前缀 vs 不共享前缀对照 | 差异同一量级（批次噪声），没有串数据 |
| GSM8K 600（`results/gsm8k600_prefixcache_184.json`） | 94.83%（不开 prefix cache：94.83%） |
| 199K 前缀 + 500 新 token 的首字时间 | 冷 7.04 s → 命中 **0.21 s** |

不命中时的 200K 单机压测（`results/ctx200k-vllm-1host-prefixcache2-20261022-102117.*`，对比 10-12 的 per-layer 基线）：

| 并发 | TTFT p50 | TPOT p50 | tok/s/chip（总 token / 墙钟） |
| --- | --- | --- | --- |
| 1 | 7.39 → 7.54 s | 6.23 → 6.76 ms | 6086 → 5946 |
| 2 | 14.19 → 14.72 s | 6.99 → 7.88 ms | 6615 → 6344 |
| 4 | 28.45 → 29.48 s | 6.96 → 7.85 ms | 6806 → 6549 |

decode 还慢 8–13%，原因未查。

两机 DP2 的 200K 压测（不命中；`results/ctx200k-vllm-dp2-*-20261022-*`，tok/s/chip 为总 token / 墙钟）：

| 并发 | TTFT p50（per-layer / prefix cache / + 调度器修正） | TPOT p90 ms | tok/s/chip |
| --- | --- | --- | --- |
| 1 | 7.40 / 7.48 / 7.53 s | 6.23 / 6.76 / 6.72 | 3041 / 2985 / 2972 |
| 2 | 7.48 / 7.55 / 8.02 s | 6.29 / 6.84 / 7.29 | 5972 / 5876 / 5270 |
| 4 | 14.17 / 14.75 / 14.60 s | 7.09 / 7.87 / 7.92 | 6617 / 6324 / 6382 |
| 8 | 28.27 / 32.78 / 29.33 s | 7.51 / 8.02 / 7.91 | 6821 / 5366 / 6564 |
| 12 | 44.73 / 63.59 / 45.63 s | 18.81 / 81.43 / 30.53 | 6502 / 4597 / 6382 |
| 16 | 57.87 / 59.93 / 59.11 s | 27.89 / 30.05 / 27.39 | 6766 / 6518 / 6620 |

- **调度器修正**（第二个 patch）：align 模式下 prefill chunk 要落在 block 边界，block 变成 2048 后，
  `SharedPrefillBudgetScheduler` 原来"预算平分再对齐到 block"的做法在每台 3、5、6 条并发时用不满 8192 的预算
  （6 条时只用 6144），c8/c12 掉 21–29%。改成按请求注水：每条请求的上限 = 剩余预算 ÷ 剩余请求数，
  对齐截掉的部分留给后面的请求。不开 prefix cache 时行为不变。
- 修正后的 c2 只有 2 个样本，落到 .184 的那条是它重启后的第一个 200K 请求（c1 的 3 条都在 .183），慢了 1 s。

**客户端（agent）注意**：
- 对话历史只追加，不改前面的内容（system prompt 里不要放时间、计数等每轮变化的内容）；末尾不满 2048 的部分每轮重算。
- 同一会话的 system 消息和第一条 user 消息保持不变，或每个会话带固定的 `X-Session-Id` 请求头，否则路由可能换机器、不命中。
- 思考内容在 `reasoning_content`；工具调用按 OpenAI 格式（`tools` / `tool_calls` / `role: tool`）。
- 每台最多 8 条并发；prompt + `max_tokens` ≤ 204800。

### 3.7 MTP 投机解码

```bash
# 在 3.6 的命令上加 spec 参数；SPEC_WARMUP=0 和更长的 RPC 超时见下面的"注意"
SPEC_WARMUP=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 MODEL=/home/zzl_google_com/models/GLM-5.3-Flash \
MAX_MODEL_LEN=204800 KV_BYTES=34359738368 bash guide/scripts/restart_serve.sh <tag> \
  --enable-prefix-caching --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# 健康后先发一个请求预热：第一个请求要编译草稿模型（K=1 约 1.5 分钟，K=3 约 9 分钟）
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","prompt":"hi","max_tokens":4}'
```

`num_speculative_tokens`（K）是每步草稿模型猜的 token 数；目标模型一次验证 K+1 个位置，猜对几个就多出几个。

**做法**（`guide/patches/0003`–`0005`；0003、0004 是上游的 #1711、#1811，pooled KDA 的逐位置状态检查点）：
- `glm5_next/mtp.py`：checkpoint 第 45 层就是 MTP 层。`enorm`/`hnorm` 后拼接 `[embed, hidden]` 过 `eh_proj`，再过一层
  sparse MLA + MoE，最后 `shared_head.norm`；embed 和 lm_head 与主模型共用。
- indexer 的 member 状态（未凑满一组 4 个的 token）改成 `kpool + K` 行的环：一次验证可能凑满一组，组里的成员最远
  来自 K 个位置之前，而被拒的草稿写的行不能覆盖它们。环从读偏移选中的检查点 block 读，写回窗口里每个检查点 block。
- 草稿模型对所有 token 桶只 trace 一次，所以：草稿里的 MLA 不走全头 SP 路径（会把 trace 特化到 token 数）；
  草稿不在 1 个 token 上编译或调用（size-1 的 guard 会把其他桶特化掉），最小补到 2。
- 调度器：一条 decode 占 `1 + K` 个 token 的预算。

**验证**（TP8；16K 上下文和 200K 线上配置）：

| 测试 | 不开投机 | K=1 | K=3 |
| --- | ---: | ---: | ---: |
| GSM8K 600（`results/gsm8k600_*.json`） | 95.33% | 94.83% | 94.83% |
| 长上下文精确检索 210 题（`longctx_recall.py`，3K–13K token） | 197 | — | 200 |
| 草稿接受率（GSM8K） | — | 0.92 | 0.74 |

- 输出和不开投机的不逐位一致：草稿/验证改变了 batch 形状，bf16 数值随之变化。合成 prompt 下一 token 分布很平，
  首个 token 会因此换成别的候选；不开投机的两种配置（unified vs per-layer）之间也有同样幅度的差异。判断对错看任务精度。
- KDA 窗口（`test_kda_spec_window.py`，K=1/3）、spec 与非 spec 的 prefill（`test_kda_spec_vs_nospec.py`）、
  member 环（`test_indexer_spec_ring.py`，环只留 kpool 行时报错）单测通过。

**速度**（200K 线上配置，单机 TP8，4 chip；结果在 `results/`）：

| 场景 | 不开投机 | K=3 |
| --- | ---: | ---: |
| 单流 decode，prompt 300 / 4000 / 32000（`decode_speed.py`） | 117 / 117 / 118 tok/s | 264 / 210 / 226 tok/s |
| 2 万进 / 700 出，每卡输出 tok/s，c1 / c2 / c4 / c8（`conc-20k-mtp-20261023.txt`） | 28.6 / 53.8 / 98.6 / 157.9 | 62.4 / 102.7 / 122.7 / 172.4 |
| 200K 进 / 128 出，TPOT p50 ms，c1 / c2 / c4 / c8（`ctx200k-1host-*-20261023-*`） | 6.83 / 7.87 / 7.95 / 10.0 | 2.73 / 3.65 / 7.38 / 11.9 |
| 200K 进 / 128 出，TTFT p50 s，c1 | 7.49 | 7.86 |

- **并发越高收益越小**：K=3 每条请求每步验证 4 个 token。c1–c2 时 decode 卡在读权重，多算几个 token 几乎不花时间；
  c4 以上 16/32 个 token 被补到 128 的编译桶（线上桶是 1,4,8,128,…），验证的计算量也随并发涨，收益基本抵消，
  200K c8 反而慢。补 16/32/64 的桶、或并发高时改 K=1 是下一步要测的。
- prefill 多跑一遍草稿，TTFT 慢 5–10%；200K 压测（99.9% 是 prefill）总吞吐低 2–9%。

**注意**：
- `SPEC_WARMUP=0`：spec 的预热编译会撞上 size-1 的 guard 失败，先关掉，改为首个请求时编译；因此要把
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 调到 1800，否则首个请求编译草稿超过默认 ~300 s 的 RPC 超时。
- 只支持 unified pool（`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`）；per-layer 布局下 indexer 的 member 状态没有检查点。
- ngram 投机要加 `--no-async-scheduling`。
- 流式压测要按 `usage.completion_tokens` 计 token：spec 下一个 SSE 事件可能带多个 token（`conc_bench.py` 已改）。

## 4. 测试脚本

脚本都在 `guide/tools/`，用 `.venv/bin/python` 运行；`--url` 指向单个实例（:8000）或路由（:8080）。

### 4.1 功能 / 精度

| 脚本             | 用途                                            | 示例                                                         |
| ---------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| `needle_test.py` | 大海捞针：在 N 条记录里埋一条访问码，问模型     | `--filler 600`（7.6K token）；`--filler 8100 --depth 0.1/0.55/0.9`（~190K） |
| `logprob_ab.py`  | 下一个 token 的 top-5 logprob A/B，比较两个配置 | `--url ... --dump ab/x.json`；`--compare ab/a.json ab/b.json` |
| `gsm8k_eval.py`  | GSM8K 零样本 chat、greedy                       | `--limit 600 --out evals/x.json`                             |
| `longctx_recall.py` | 长上下文精确检索（3K–13K token），多台同题比准确率 | `--url ref=http://a:8000 --url mtp=http://b:8000 --n 150` |
| `spec_ab.py`     | 同一 greedy 请求打 spec / 非 spec 两台，比 token、接受率 | `--spec http://b:8000 --ref http://a:8000 --lengths 300,4000` |
| `first_token_check.py` / `spec_tf_check.py` | A 生成的 token 落在参考 top-5 的哪里（首 token / 逐位置） | `--a http://b:8000 --ref http://a:8000` |
| `test_kda_spec_*.py` / `test_indexer_spec_ring.py` | MTP 用到的 KDA 窗口、member 环单测（需要空闲 TPU） | `K=3 python ...`、`RING=7 python ...` |

### 4.2 性能

| 脚本                                 | 用途                                                         | 示例                                  |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------- |
| `ctx200k_bench.py`                   | **200K 进 / 128 出压测**，并发梯度 1/2/4/8/12/16，指标口径与 GPU 基线逐字一致，输出 JSON + TSV | 见下                                  |
| `ctx200k_true_tput.py`               | 从 ctx200k JSON 算真实吞吐（总 token / 墙钟时间）            | `ctx200k_true_tput.py results/x.json` |
| `ttft_bench.py`                      | 单流：不同 prompt 长度的 TTFT 和 decode tok/s                | `--fillers 3,20,60,120,300,1200`      |
| `conc_bench.py`                      | 并发压测（`--filler 340` ≈ 7.6K、`900` ≈ 2 万 token），按 usage 计 token |  `--filler 900 --out 700 --conc 1,2,4,8` |
| `decode_speed.py`                    | 单流 decode tok/s（不流式、无 logprobs，256 与 1 token 的时间差） | `--url ref=http://a:8000 --url mtp=http://b:8000` |

200K 压测命令：

```bash
cd /zzlfs/glm53flash_torchtpu
ts=$(date +%Y%m%d-%H%M%S)
.venv/bin/python guide/tools/ctx200k_bench.py --base http://127.0.0.1:8080 \
  --model GLM-5.3-Flash --chips 8 --warmup --out results/ctx200k-vllm-dp2-$ts.json
#   单机：--base http://127.0.0.1:8000 --chips 4；只跑部分并发：--ladder 1,2,4
.venv/bin/python guide/tools/ctx200k_true_tput.py results/ctx200k-vllm-dp2-$ts.json
```

注意：TSV 的 Framework / Topo 两列是从 `/get_server_info` 推的，对本服务会误标成 sglang-jax，发表前要手工改
（例如 `guide/results/ctx200k-vllm-dp2-spmla-20261012-065404-labeled.tsv`）。

### 4.3 Profile

服务启动时加 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>","torch_profiler_with_stack":true,"torch_profiler_use_gzip":true}'`，然后：

| 脚本                                                         | 用途                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| `prof_prefill.py --url ... --prefix p100k --filler 4400 --delay 4.0` | 抓长 prompt 末尾（~100K 处）的几个 prefill 步       |
| `prof_decode.py --url ... --prefix dec --filler 120 --steps 20` | 抓 decode 步                                        |
| `prof_window.py <trace.gz> --index N`                        | 单个 XLA module（一步）内按来源 / op 统计 self time |
| `prof_summary.py <trace.gz> --steps N [--shapes copy]`       | 每步 op 汇总，可按 HLO 签名拆分（查布局转换 copy）  |
| `prof_diff.py <trace_a.gz> <trace_b.gz>`                     | 两个 trace 中单步内各 op 的每次调用耗时对比（定位回退）  |

### 4.4 单卡 / 多卡对拍（需要空闲 TPU）

| 脚本                                         | 内容                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| `test_mla_token_shard.py`                    | sparse MLA 按 token 切片 vs 整批（逐位比较）+ 单卡耗时       |
| `test_mla_heads.py`                          | sparse MLA kernel 8 头 vs 64 头逐头一致                      |
| `test_mla_token_shard_mesh.py`               | 8 device mesh 纯 JAX：token 切分算子、本卡 token 算子 vs 单卡参考（含 all-to-all） |
| `test_indexer_token_shard.py`                | indexer 按 token 切片选择 vs 整批（含 decode + prefill 混排） |
| `test_indexer_v2.py` / `bench_indexer_v2.py` | v1 vs v3 indexer 对拍 / 单层耗时                             |
| `bench_moe_permute.py`                       | MoE 分发 / 合并：XLA gather vs SparseCore kernel             |
| `repro_sc_gather_offload.py`                 | 纯 JAX 复现 libtpu 0.0.50 的 SparseCore gather 回退（见第 7 节第 4 条） |

---

## 5. 结果

### 5.1 200K 进 / 128 出，DP2 × TP8 + SP（8 chip，最新版本）

数据：`guide/results/ctx200k-vllm-dp2-spmla-20261012-065404.*`，45/45 请求成功。上面 6 行是同硬件的 sglang-jax 对照。

| Model             | Framework     | Topo                      | Context Length | Requests | Concurrency | Input/Output | TTFT p50 (ms) | TTFT p90 (ms) | TPOT p50 (ms/tok) | TPOT p90 (ms/tok) | Extend tput p50 (tok/s) | Decode tput p50 (tok/s) | tok/s/chip | E2E tput p50 (tok/s) | E2E tput p90 (tok/s) | E2E latency p50 (s) | E2E latency p90 (s) |
| :---------------- | :------------ | :------------------------ | :------------- | -------: | ----------: | :----------- | ------------: | ------------: | ----------------: | ----------------: | ----------------------: | ----------------------: | ---------: | -------------------: | -------------------: | ------------------: | ------------------: |
| GLM-5.3-Flash-FP8 | sglang-jax    | 8×TPUv7x TP16/EP16 DP1+SP | 200K           |        3 |           1 | 199699/128   |          7410 |          7506 |             12.52 |             12.52 |                   26951 |                    79.9 |     2771.9 |                22175 |                22205 |                9.01 |                9.11 |
| GLM-5.3-Flash-FP8 | sglang-jax    | 8×TPUv7x TP16/EP16 DP1+SP | 200K           |        2 |           2 | 199700/128   |         13979 |         13981 |             14.72 |             14.72 |                   14286 |                      68 |     3149.4 |                12598 |                12599 |               15.86 |               15.86 |
| GLM-5.3-Flash-FP8 | sglang-jax    | 8×TPUv7x TP16/EP16 DP1+SP | 200K           |        4 |           4 | 199700/128   |         27564 |         27569 |             18.96 |             18.96 |                    7245 |                    52.7 |     3331.6 |                 6663 |                 6665 |               29.99 |                  30 |
| GLM-5.3-Flash-FP8 | sglang-jax    | 8×TPUv7x TP16/EP16 DP1+SP | 200K           |        8 |           8 | 199700/128   |         53906 |         53918 |             14.95 |             14.96 |                    3705 |                    66.9 |     3579.8 |                 3580 |                 3581 |               55.82 |               55.83 |
| GLM-5.3-Flash-FP8 | sglang-jax    | 8×TPUv7x TP16/EP16 DP1+SP | 200K           |       12 |          12 | 199700/128   |         80298 |         80315 |             15.61 |             15.62 |                    2487 |                      64 |     3642.2 |                 2428 |                 2429 |                82.3 |               82.31 |
| GLM-5.3-Flash-FP8 | sglang-jax    | 8×TPUv7x TP16/EP16 DP1+SP | 200K           |       16 |          16 | 199700/128   |        106169 |        106193 |             16.19 |             16.21 |                    1881 |                    61.8 |     3692.3 |                 1846 |                 1847 |              108.24 |              108.27 |
| GLM-5.3-Flash-FP8 | vllm-torchtpu | 8×TPUv7x DP2×TP8+SP       | 200K           |        3 |           1 | 199700/128   |          7455 |          7460 |              6.14 |              6.15 |                   26788 |                   162.8 |     3031.0 |                24248 |                24323 |                8.24 |                8.25 |
| GLM-5.3-Flash-FP8 | vllm-torchtpu | 8×TPUv7x DP2×TP8+SP       | 200K           |        2 |           2 | 199700/128   |          7739 |          7981 |              6.58 |              6.85 |                   25843 |                   152.3 |     5830.8 |                23323 |                24072 |                8.58 |                8.86 |
| GLM-5.3-Flash-FP8 | vllm-torchtpu | 8×TPUv7x DP2×TP8+SP       | 200K           |        4 |           4 | 199700/128   |         14192 |         14224 |              7.11 |              7.12 |                   14071 |                   140.7 |     6616.0 |                13232 |                13260 |               15.10 |               15.13 |
| GLM-5.3-Flash-FP8 | vllm-torchtpu | 8×TPUv7x DP2×TP8+SP       | 200K           |        8 |           8 | 199699/128   |         28247 |         28335 |              7.47 |              7.48 |                    7070 |                   133.8 |     6843.1 |                 6843 |                 6854 |               29.20 |               29.29 |
| GLM-5.3-Flash-FP8 | vllm-torchtpu | 8×TPUv7x DP2×TP8+SP       | 200K           |       12 |          12 | 199700/128   |         44713 |         44809 |              9.46 |             18.64 |                    4466 |                   105.7 |     6526.6 |                 4351 |                 4361 |               45.93 |               46.00 |
| GLM-5.3-Flash-FP8 | vllm-torchtpu | 8×TPUv7x DP2×TP8+SP       | 200K           |       16 |          16 | 199700/128   |         57849 |         57997 |              8.66 |             27.90 |                    3452 |                   115.5 |     6779.9 |                 3390 |                 3397 |               58.95 |               59.11 |

按「总 token / 墙钟时间」算的真实吞吐（tok/s/chip）：c1 3032、c2 5590、c4 6594、c8 6816、c12 6510、c16 6755。

### 5.2 优化过程（200K）

| 版本                      | 主要改动                                                     | 100K 处 8K chunk | 单机 c1 TTFT | DP2 饱和真实 tok/s/chip |
| ------------------------- | ------------------------------------------------------------ | ---------------: | -----------: | ----------------------: |
| v3                        | 起点（indexer v2/v3、调度器、member 每请求状态）             |           504 ms |       16.1 s |                   ~3150 |
| + token 切分              | sparse MLA 和 indexer 按 token 切给 8 卡（每卡 1/8 token × 64 头） |           376 ms |       9.74 s |                   ~5100 |
| + SP                      | 残差流按 token 切，mHC 52 -> 5 ms，通信 62 -> 40 ms          |          ~295 ms |       8.28 s |                   ~6050 |
| + 全头 DSA / 跳过 rope 写 | DSA 层免掉 all-to-all 和 reduce-scatter；GLM 恒 0 的 rope cache 不再每步写 |   未单独 profile |   **7.39 s** |               **~6800** |

### 5.3 单机单流（`ttft_bench.py`，.38，最新版本）

| prompt tok | TTFT s | decode tok/s |
| ---------: | -----: | -----------: |
|      1,208 |  0.094 |          124 |
|      5,878 |  0.286 |          128 |
|     23,478 |  1.029 |          128 |

`GLM53_SP_ALL_HEADS=0` 时：5,878 tok TTFT 0.330 s、23,478 tok 1.167 s，decode ~139 tok/s。

### 5.4 精度

| 配置               | GSM8K 600 | needle                             |
| ------------------ | --------: | ---------------------------------- |
| TP 基线（v3）      |    94.83% | —                                  |
| MoE EP             |    95.50% | —                                  |
| + token 切分       |    95.00% | 2.6K / 7.6K / 190K（3 个深度）全中 |
| + SP               |    95.67% | 全中                               |
| + 全头 DSA（当前） |    95.17% | 全中                               |

标准误约 0.9%，各版本之间无显著差异。新算子在单卡和 8 device mesh 上与参考实现逐位相同。
logprob A/B：token 切分只在 512–2048 桶有数值级差异（max 1.0）；SP 之后全桶有差异（max ~2），
推测是 reduce-scatter 替换 all-reduce 改变了 bf16 求和顺序（每次 forward 90 次规约），GSM8K / needle 无退化。

---

## 6. 指标定义（`ctx200k_bench.py`，与 GPU 基线脚本逐字一致）

- **TTFT**：第一个非空 content 分片到达的时间。
- **gen** = e2e − TTFT；**TPOT** = gen / completion_tokens；**Decode tput** = completion_tokens / gen（与 TPOT 互为倒数）。
- **Extend tput** = prompt_tokens / TTFT；**E2E tput** = (prompt + completion) / e2e。
- **tok/s/chip** = p50(单请求 E2E tput) × 并发 / 芯片数。请求排队时会虚高，真实吞吐用 `ctx200k_true_tput.py`。
- 百分位为线性插值；每条 prompt 带唯一 nonce（冷前缀）；c1 发 3 条，其余档 n = 并发。

---

## 7. 已知问题与注意事项

1. **decode 取舍**：`GLM53_SP_ALL_HEADS=1` 让 decode 也读完整的 q_b/o_proj（为了让 decode 图和 prefill 图读同一组权重，
   runner 要求两份 trace 输入签名一致），单流 decode 慢 ~8%。decode 为主时设 0。
   
2. **高并发 TPOT p90**：每副本最多 8 条并发，c12/c16 时部分请求的 decode 会陪着别的请求的 8K prefill chunk 跑，TPOT p90 升到 19–28 ms。

3. **冷启动 ~35 分钟**：主要是 9 个 token 桶的编译（SP 让 >= 128 的桶多一次 trace）。

4. **不要用 libtpu 0.0.50**：它的 SparseCore gather 下放比 0.0.47 慢（单个 SparseCore 慢 43%），MoE 分发/合并的
   gather 每层慢 28–41%，200K prefill 整体慢 ~3%（c1 TTFT 7.39 -> 7.64 s，饱和吞吐 6806 -> 6587 tok/s/chip）。
   没有 libtpu flag 能绕过。复现：`guide/tools/repro_sc_gather_offload.py`（0.0.47 0.588 ms vs 0.0.50 0.740 ms）。

5. **Prefix cache（3.6 节）**：不命中时 decode 比 per-layer 布局慢 8–13%（未查），200K 吞吐低 2–4%；命中粒度 2048 token；`chat_template_kwargs: {"enable_thinking": false}` 关不掉思考（思考会进 `content`）。
   prefix cache、调度器和 MTP 的改动已推到 PR #1859 的分支，但 PR 分支钉的 vLLM 是 `8039d3cb`，
   和 `.venv` 的 vLLM 0.29.0 不兼容，只在 `glm53-port`（vLLM 0.29.0）上验证过。

   

6. **MTP（3.7 节）**：首个请求编译草稿（K=3 约 9 分钟），上线前先预热；c4 以上收益很小，200K c8 比不开慢；
   vLLM 的按 batch 选 K（`num_speculative_tokens_per_batch_size`）插件还不支持。
