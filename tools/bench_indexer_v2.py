"""Time the indexer op per layer call, v1 (XLA) vs v2 (streamindex_topk), on TPU.

    source vllm_env.sh; python bench_indexer_v2.py [--max-len 8192] [--ctx 7000]
        [--tokens 4] [--seqs 4] [--skip-v1]

Decode: `--tokens == --seqs`, one token per sequence. Prefill: one sequence
taking `--tokens` new tokens on top of `--ctx - tokens` already cached.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import sys

sys.path.insert(0, "/zzlfs/glm53flash_torchtpu/vllm-torchtpu/src")
from vllm_torchtpu.models.vllm.glm5_next.indexer import (  # noqa: E402
    _kpool_index_jax, _kpool_index_v3_jax)

ap = argparse.ArgumentParser()
ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--ctx", type=int, default=7000)
ap.add_argument("--tokens", type=int, default=4)
ap.add_argument("--seqs", type=int, default=4)
ap.add_argument("--max-seqs", type=int, default=4)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--skip-v1", action="store_true")
a = ap.parse_args()

BS, HD, HEADS, KPOOL, SELECT_K = 256, 128, 32, 4, 512
WIDTH = 2304
pages_per_seq = a.max_len // BS
num_blocks = a.max_seqs * pages_per_seq + 8
rng = np.random.default_rng(0)
decode = a.tokens == a.seqs

seq_lens = np.zeros(a.max_seqs, np.int32)
cu = np.zeros(a.max_seqs + 1, np.int32)
if decode:
    seq_lens[: a.seqs] = a.ctx
    cu[1 : a.seqs + 1] = np.arange(1, a.seqs + 1)
    cu[a.seqs + 1 :] = a.seqs
    dist = np.array([a.seqs, a.seqs, a.seqs], np.int32)
else:
    seq_lens[0] = a.ctx
    cu[1:] = a.tokens
    dist = np.array([0, 0, 1], np.int32)
block_table = rng.permutation(num_blocks)[: a.max_seqs * pages_per_seq].astype(np.int32)

args = (
    jnp.asarray(rng.standard_normal((a.tokens, HD)), jnp.bfloat16),
    jnp.asarray(rng.standard_normal((a.tokens, HD)), jnp.bfloat16),
    jnp.asarray(rng.standard_normal((a.tokens, HEADS, HD)), jnp.float32),
    jnp.asarray(rng.uniform(0, 1, (a.tokens, HEADS)), jnp.float32),
    jnp.asarray(rng.standard_normal((KPOOL, HD)), jnp.float32),
    jnp.asarray(seq_lens), jnp.asarray(block_table), jnp.asarray(cu),
)


def bench(fn, caches):
    caches = fn(*caches)[:-1]
    jax.block_until_ready(caches)
    t = time.perf_counter()
    for _ in range(a.iters):
        caches = fn(*caches)[:-1]
    jax.block_until_ready(caches)
    return (time.perf_counter() - t) / a.iters * 1e3


member = lambda: jnp.asarray(  # noqa: E731
    rng.standard_normal((num_blocks, BS // 2, 2, 2 * HD)), jnp.bfloat16)
label = (f"{'decode' if decode else 'prefill'} tokens={a.tokens} ctx={a.ctx} "
         f"max_len={a.max_len}")
if not a.skip_v1:
    v1 = jax.jit(lambda c: _kpool_index_jax(
        c, *args, kpool=KPOOL, select_k=SELECT_K, width=WIDTH,
        max_pools=a.max_len // KPOOL, num_heads=HEADS), donate_argnums=(0,))
    print(f"v1 {label}: {bench(v1, (member(),)):8.3f} ms/call")
v3_args = args[:6] + (args[7], jnp.asarray(dist),
                      jnp.arange(1, a.max_seqs + 1, dtype=jnp.int32),
                      jnp.asarray(block_table))
v2 = jax.jit(lambda c, kc: _kpool_index_v3_jax(
    c, kc, *v3_args, kpool=KPOOL, select_k=SELECT_K, width=WIDTH),
    donate_argnums=(0, 1))
key_cache = jnp.asarray(
    rng.integers(0, 64, (num_blocks, BS // KPOOL // 4, 4, 256)), jnp.uint8)
state = jnp.zeros((a.max_seqs + 1, KPOOL, 2 * HD), jnp.bfloat16)
print(f"v3 {label}: {bench(v2, (state, key_cache)):8.3f} ms/call")
