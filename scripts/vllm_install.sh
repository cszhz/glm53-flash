#!/usr/bin/env bash
# 为 GLM-5.3-Flash 的 torch+vLLM 路线新建独立 venv。
# 不动 /zzlfs/glm53flash/.venv（旧的 editable 指向已删除的 checkout），
# 也不动 glm53flash 下已有的 sgl-jax 脚本。
#
# 鉴权：私有源走 GAR keyring -> ADC。gcloud CLI 的活跃账号是 compute SA，
# 对 torch-tpu-virtual-registry 是 403；有权限的是 ADC 里的 authorized_user。
# 所以只要 ~/.config/gcloud/application_default_credentials.json 是
# authorized_user 且没过期就行，不需要切 gcloud 账号。
set -Eeuo pipefail

PROJ=/zzlfs/glm53flash_torchtpu
VENV="${VENV:-$PROJ/.venv}"
SRC="${SRC:-$PROJ/vllm-torchtpu}"
PY="${PY:-python3.12}"

export PIP_INDEX_URL="https://oauth2accesstoken@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
# vllm-torchtpu 是 out-of-tree TPU platform，vLLM 本体从源码建但不编 device kernel。
export VLLM_TARGET_DEVICE=empty

if [[ ! -d "$VENV" ]]; then
  echo "=== 建 venv: $VENV"
  "$PY" -m venv "$VENV" --symlinks
  # 先有鸡还是先有蛋：keyring 本身要装，但私有源恰恰需要 keyring 才能鉴权。
  # 所以 bootstrap 这一步必须显式走公共 PyPI。
  PIP_INDEX_URL="https://pypi.org/simple" \
    "$VENV/bin/pip" install -q --upgrade pip
  PIP_INDEX_URL="https://pypi.org/simple" \
    "$VENV/bin/pip" install -q keyring keyrings.google-artifactregistry-auth
fi

cd "$SRC"
echo "=== HEAD: $(git log --oneline -1)"
echo "=== 开始安装 $(date +%H:%M:%S)"
"$VENV/bin/pip" install --pre -e ".[benchmarking]" --group vllm
# pyproject 把 transformers 钉在 <5.15（为 vllm-omni），但 glm5_next 的 config 类
# 5.16 才进 transformers。我们不用 omni，单独抬到旧环境验证过的 5.17.0。
"$VENV/bin/pip" install -q "transformers==5.17.0"
# libtpu 统一用 0.0.47：0.0.50 的 SparseCore gather 下放慢（MoE 分发/合并 gather 每层 +28–41%，200K prefill
# 慢 ~3%），见 guide/Test_Guide.md 第 7 节。从钉 0.0.50 的 checkout（如 PR 分支）安装时这一步把它换回来。
"$VENV/bin/pip" install -q "libtpu==0.0.47" "tpu-sync-torch==0.0.1.dev20261002213151"
"$VENV/bin/python" -c "import importlib.metadata as m; print('=== libtpu', m.version('libtpu'))"
echo "=== 安装完成 $(date +%H:%M:%S)"

echo "=== 关键版本："
"$VENV/bin/pip" list 2>/dev/null | grep -iE "^(vllm|vllm-torchtpu-core|vllm-torchtpu|vllm_torchtpu|torch|torch-tpu|torch_tpu|torchvision|libtpu|jax|jaxlib|tpu-raiden-torch|transformers) "
