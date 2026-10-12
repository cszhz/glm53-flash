"""Does the sparse MLA kernel give each head the same output at 8 and 64 heads?

    source vllm_env.sh; SRC=<worktree> python test_mla_heads.py

One request, prefill from position 0 (so kv_len = pos + 1 < topk for short
prompts), attended once with 64 heads and once as 8 groups of 8 heads.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.environ.get(
    "SRC", "/zzlfs/glm53flash_torchtpu/vllm-torchtpu") + "/src")
from vllm_torchtpu.kernels.mla.sparse.kernel import (  # noqa: E402
    sparse_ragged_paged_attention)

NOPE, ROPE, TOPK, BS = 512, 64, 2304, 256
rng = np.random.default_rng(0)
for T, real, ctx0 in ((512, 240, 0), (1024, 540, 0), (2048, 1550, 0),
                      (4096, 3050, 0), (128, 60, 0), (1024, 540, 5000)):
    pages = -(-(ctx0 + T) // BS) + 1
    nope = jnp.asarray(rng.integers(0, 120, (pages, BS, 4, 128)), jnp.uint8)
    rope = jnp.asarray(rng.integers(0, 120, (pages, BS // 4, 4, 128)), jnp.uint8)
    topk = np.full((T, TOPK), -1, np.int32)
    for r in range(real):
        vis = ctx0 + r + 1
        k = min(2051, vis)
        topk[r, :k] = np.sort(rng.choice(vis, k, replace=False))
    q = jnp.asarray(rng.standard_normal((T, 64, NOPE + ROPE)), jnp.bfloat16)
    args = (jnp.asarray(topk), jnp.arange(pages, dtype=jnp.int32),
            jnp.asarray([0, real], jnp.int32), jnp.asarray([0, 0, 1], jnp.int32))
    f = jax.jit(lambda q, tk, pi, cu, d: sparse_ragged_paged_attention(
        q, nope, rope, tk, pi, cu, d, sm_scale=0.06))
    o64 = np.asarray(f(q, *args), np.float32)[:real]
    o8 = np.concatenate([np.asarray(f(q[:, g * 8:(g + 1) * 8], *args), np.float32)
                         for g in range(8)], axis=1)[:real]
    d = np.abs(o64 - o8).max(axis=(1, 2))
    print(f"T={T} real={real} ctx0={ctx0}: max|diff| {d.max():.3g}, "
          f"rows differing {(d > 0).sum()} (first {np.argmax(d > 0) if (d > 0).any() else '-'})")
