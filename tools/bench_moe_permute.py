"""TP MoE token movement at prefill shape: XLA gather vs SparseCore kernels.

    source vllm_env.sh; SRC=<worktree> python bench_moe_permute.py [--tokens 8192]

Dispatch: `hidden[token_indices_sorted]` (what the TP path does) against
`ragged_gather_v2`. Combine: the TP path's gather + weight + sum against
`ragged_gather_reduce` v1 and v3. Outputs are checked against the XLA ones.
"""
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.environ.get(
    "SRC", "/zzlfs/glm53flash_torchtpu/vllm-torchtpu") + "/src")
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import (  # noqa: E402
    ragged_gather_reduce as rgr_v1)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (  # noqa: E402
    ragged_gather_reduce as rgr_v3)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import (  # noqa: E402
    ragged_gather_v2)

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=8192)
ap.add_argument("--hidden", type=int, default=4096)
ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--experts", type=int, default=288)
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
T, H, K, E = a.tokens, a.hidden, a.topk, a.experts
rng = np.random.default_rng(0)

x = jnp.asarray(rng.standard_normal((T, H)), jnp.bfloat16)
ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)]).astype(np.int32)
order = np.argsort(ids.reshape(-1), kind="stable").astype(np.int32)
tok_sorted = jnp.asarray(order // K)
revert = jnp.asarray(np.argsort(order).astype(np.int32))
w = jnp.asarray(rng.uniform(0, 1, T * K), jnp.float32)
valid = jnp.ones(T * K, bool)
y = jnp.asarray(rng.standard_normal((T * K, H)), jnp.bfloat16)  # gmm2 output


def timeit(f, *args):
    out = f(*args)
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(a.iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / a.iters * 1e3, out


ms, ref = timeit(jax.jit(lambda x, i: x[i]), x, tok_sorted)
print(f"dispatch XLA gather        {ms:7.2f} ms")
ms, got = timeit(jax.jit(lambda x, i: ragged_gather_v2(
    x, i, jnp.array([0], jnp.int32), jnp.array([T * K], jnp.int32))), x, tok_sorted)
print(f"dispatch ragged_gather_v2  {ms:7.2f} ms  equal={bool((got == ref).all())}")


@jax.jit
def combine_xla(y, rev, w, valid):
    h = y[rev].reshape(T, K, H) * w.reshape(T, K)[..., None]
    h = jnp.where(valid.reshape(T, K)[..., None], h, 0)
    return h.sum(axis=1).astype(y.dtype)


ms, ref = timeit(combine_xla, y, revert, w, valid)
print(f"combine XLA gather+sum     {ms:7.2f} ms")
for name, fn in (("v1", rgr_v1), ("v3", rgr_v3)):
    try:
        ms, got = timeit(jax.jit(lambda y, r, w, v, fn=fn: fn(
            y, r, w, v, reduce_group_size=K).astype(y.dtype)), y, revert, w, valid)
        d = float(jnp.abs(got.astype(jnp.float32) - ref.astype(jnp.float32)).max())
        print(f"combine ragged_gather_reduce {name} {ms:7.2f} ms  max|diff| {d:.3g}")
    except Exception as e:  # noqa: BLE001
        print(f"combine ragged_gather_reduce {name} failed: {type(e).__name__}: {str(e)[:200]}")
