#!/usr/bin/env bash
# 在本机（重）起同目录的 vllm_serve.sh，日志写 /zzlfs/glm53flash_torchtpu/logs/<tag>.log，立刻返回。远程用：
#   ssh 10.128.0.38 'bash /zzlfs/glm53flash_torchtpu/guide/scripts/restart_serve.sh <tag> [serve 参数...]'
# 环境变量（MAX_MODEL_LEN、KV_BYTES、...）照常透传给 vllm_serve.sh。
set -u
TAG=${1:?usage: restart_serve.sh <tag> [args...]}; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /zzlfs/glm53flash_torchtpu
pkill -INT -f "[v]llm.entrypoints.openai.api_server"
for _ in $(seq 1 60); do pgrep -f "[v]llm.entrypoints|[E]ngineCore|[W]orker_TP" >/dev/null || break; sleep 3; done
# A server still starting up ignores SIGINT and its workers keep the chips
# (the next start then dies in "PjRtClient is not initialized"): kill the
# whole process groups.
for p in $(pgrep -f "[v]llm.entrypoints.openai.api_server"); do
  pkill -KILL -g "$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null
done
sleep 5
# vllm_env.sh points inductor at /dev/shm and the compile cache is off, so its
# artifacts are never reused -- but they pile up across restarts (300 GiB seen)
# and squeeze the page cache that keeps the 306 GiB of weights loadable.
rm -rf /dev/shm/torchinductor/*
setsid nohup bash "$HERE/vllm_serve.sh" "$@" > "logs/$TAG.log" 2>&1 < /dev/null &
echo "started $TAG on $(hostname -I | awk '{print $1}'), pid $!"
