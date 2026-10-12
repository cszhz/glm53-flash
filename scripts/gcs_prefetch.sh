#!/usr/bin/env bash
# 把 /gcs 上的权重并行读进 page cache，之后 vLLM 加载就不再受 gcsfuse 单流限速。
#
#   bash /zzlfs/glm53flash_torchtpu/gcs_prefetch.sh                      # 默认 GLM-5.3-Flash
#   DIR=/gcs/models/xxx JOBS=64 bash gcs_prefetch.sh
#
# 为什么需要这个：gcsfuse 单个文件顺序读只有 ~40 MB/s，306 GiB 要两个小时。
# 并行到几十路之后能吃满网卡。机器 944 GiB 内存，306 GiB 全放得下，读完之后
# 同一台机器上后续每次起服务都是内存命中，不再走网络。
#
# vLLM 自己的 --safetensors-load-strategy=prefetch 也能并行读，但它是**每个 TP
# worker 各跑一遍**，8 个 worker 同时抢同一批文件，不如在起服务之前读一次干净。
set -Eeuo pipefail

DIR="${DIR:-/gcs/models/GLM-5.3-Flash}"
JOBS="${JOBS:-64}"

echo "=== prefetch $DIR  jobs=$JOBS  host $(hostname -I | awk '{print $1}')"
before=$(awk '/ens5:/ {print $2}' /proc/net/dev)
start=$(date +%s)

# -print0/-0 是为了容忍文件名里的空格；只读 safetensors 和 json，跳过 README 之类。
find "$DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.json' \) -print0 \
  | xargs -0 -r -P "$JOBS" -I{} sh -c 'cat "$1" > /dev/null' _ {}

end=$(date +%s)
after=$(awk '/ens5:/ {print $2}' /proc/net/dev)
secs=$(( end - start ))
gib=$(( (after - before) / 1024 / 1024 / 1024 ))
echo "=== 读了 ${gib} GiB，用时 ${secs}s（$(( secs > 0 ? gib * 1024 / secs : 0 )) MiB/s）"
free -g | sed -n '1,2p'
