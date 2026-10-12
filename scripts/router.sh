#!/usr/bin/env bash
# (Re)start the DP router on :8080 in front of the given backends.
#   bash router.sh http://127.0.0.1:8000 http://10.128.0.38:8000
# 路由脚本用同目录上一级的 tools/dp_router.py；日志写 /zzlfs/glm53flash_torchtpu/logs/router.log。
GUIDE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd /zzlfs/glm53flash_torchtpu
for p in $(pgrep -f "tools/dp_router.py"); do kill "$p"; done
sleep 1
args=()
for b in "$@"; do args+=(--backend "$b"); done
setsid nohup .venv/bin/python "$GUIDE/tools/dp_router.py" --port 8080 "${args[@]}" \
  > logs/router.log 2>&1 < /dev/null &
sleep 2
curl -s localhost:8080/router/stats; echo
