"""Minimal repro: XLA row gather offloaded to SparseCore, libtpu 0.0.47 vs 0.0.50.

    python repro_sc_gather_offload.py                 # default (2 SparseCores)
    LIBTPU_INIT_ARGS=--xla_tpu_num_sparse_cores_for_gather_offloading=1 \
        python repro_sc_gather_offload.py

Pure JAX, one TPU v7x device. The shape is a MoE token dispatch at prefill:
8192 tokens x top-8 experts = 65536 rows of a [8192, 4096] bf16 table, rows
sorted by expert. Measured (TPU v7x, ms per gather):

                     libtpu 0.0.47   libtpu 0.0.50
    2 SparseCores        0.59            0.75   (+27%)
    1 SparseCore         0.97            1.39   (+43%)
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

TOKENS, HIDDEN, TOPK, EXPERTS, ITERS = 8192, 4096, 8, 288, 50

rng = np.random.default_rng(0)
table = jnp.asarray(rng.standard_normal((TOKENS, HIDDEN)), jnp.bfloat16)
ids = np.stack([rng.choice(EXPERTS, TOPK, replace=False) for _ in range(TOKENS)])
rows = jnp.asarray(np.argsort(ids.reshape(-1), kind="stable") // TOPK, jnp.int32)

gather = jax.jit(lambda t, i: t[i])
jax.block_until_ready(gather(table, rows))
start = time.perf_counter()
for _ in range(ITERS):
    out = gather(table, rows)
jax.block_until_ready(out)
ms = (time.perf_counter() - start) / ITERS * 1e3
print(f"jax {jax.__version__}, {jax.devices()[0].device_kind}: "
      f"gather [{TOKENS},{HIDDEN}] bf16 -> [{rows.shape[0]},{HIDDEN}]: {ms:.3f} ms")
