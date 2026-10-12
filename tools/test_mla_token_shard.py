"""Token-sharded sparse MLA, single-device check and per-device cost model.

    source vllm_env.sh; python test_mla_token_shard.py [--shards 8]

Correctness: a ragged 3-request batch (padded) attended once with all heads
the normal way, and again as `--shards` token slices through the kernel's
`prepared_kv_lens` path (what each rank runs after the all-to-all). Rows of
real tokens must match.

Cost: one 8192-token chunk at 100K context as one TP8 rank sees it today
(8192 tokens x 8 heads) vs token-sharded (1024 tokens x 64 heads). The two
all-to-alls are not on one device and are not timed here.
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
from vllm_torchtpu.kernels.mla.sparse.kernel import (  # noqa: E402
    prepare_gather_indices, sparse_ragged_paged_attention)

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=8)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--skip-perf", action="store_true")
a = ap.parse_args()

HEADS, NOPE, ROPE, TOPK, BS = 64, 512, 64, 2304, 256
rng = np.random.default_rng(0)


def caches(pages):
    nope = jnp.asarray(rng.integers(0, 120, (pages, BS, 4, 128)), jnp.uint8)
    rope = jnp.asarray(rng.integers(0, 120, (pages, BS // 4, 4, 128)), jnp.uint8)
    return nope, rope


# --- correctness -------------------------------------------------------------
T, n = 1024, a.shards
q_lens = [300, 1, 699]  # real tokens 1000, padded to 1024
ctx = [5000, 40000, 3000]  # tokens already in cache before this step
pages_per_seq = -(-max(c + l for c, l in zip(ctx, q_lens)) // BS)
nseq = len(q_lens)
page_indices = rng.permutation(nseq * pages_per_seq).astype(np.int32)
cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
dist = np.array([0, 0, nseq], np.int32)
topk = np.full((T, TOPK), -1, np.int32)
for s in range(nseq):
    for r in range(q_lens[s]):
        visible = ctx[s] + r + 1
        k = min(TOPK - 3, visible)
        sel = np.sort(rng.choice(visible, k, replace=False))
        topk[cu[s] + r, :k] = sel
nope, rope = caches(nseq * pages_per_seq)
q = jnp.asarray(rng.standard_normal((T, HEADS, NOPE + ROPE)), jnp.bfloat16)
args = (jnp.asarray(topk), jnp.asarray(page_indices), jnp.asarray(cu),
        jnp.asarray(dist))

ref = jax.jit(lambda q, tk, pi, cu, d: sparse_ragged_paged_attention(
    q, nope, rope, tk, pi, cu, d, sm_scale=0.06))(q, *args)


@jax.jit
def sharded(q, tk, pi, cu, d):
    kv_lens, phys = prepare_gather_indices(
        tk, pi, cu, page_size=BS, total_num_pages=nope.shape[0])
    lt = T // n
    outs = []
    for s in range(n):
        start = s * lt
        outs.append(sparse_ragged_paged_attention(
            q[start:start + lt], nope, rope, phys[start:start + lt], None, cu, d,
            sm_scale=0.06, prepared_kv_lens=kv_lens[start:start + lt],
            prepared_token_end=jnp.clip(cu[d[2]] - start, 0, lt)))
    return jnp.concatenate(outs)


got = sharded(q, *args)
real = int(cu[-1])
diff = jnp.abs(got[:real].astype(jnp.float32) - ref[:real].astype(jnp.float32))
print(f"correctness: {real} real rows, max |diff| {float(diff.max()):.3g}, "
      f"rows differing {int((diff.max(axis=(1, 2)) > 0).sum())}")

# --- per-device cost ---------------------------------------------------------
if a.skip_perf:
    sys.exit(0)


def timeit(f, *args):
    jax.block_until_ready(f(*args))
    t = time.perf_counter()
    for _ in range(a.iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / a.iters * 1e3


CTX, TOK = 100_000, 8192
sp_pages = -(-CTX // BS) + 1
sp_nope, sp_rope = caches(sp_pages)
sp_topk = np.sort(rng.integers(0, CTX - TOK, (TOK, TOPK)), axis=1).astype(np.int32)
pi = jnp.arange(sp_pages, dtype=jnp.int32)
cu1 = jnp.asarray([0, TOK], jnp.int32)
d1 = jnp.asarray([0, 0, 1], jnp.int32)

q8 = jnp.asarray(rng.standard_normal((TOK, HEADS // n, NOPE + ROPE)), jnp.bfloat16)
base = jax.jit(lambda q, tk: sparse_ragged_paged_attention(
    q, sp_nope, sp_rope, tk, pi, cu1, d1, sm_scale=0.06))
ms_base = timeit(base, q8, jnp.asarray(sp_topk))
print(f"head-sharded  ({TOK} tok x {HEADS // n} heads): {ms_base:7.2f} ms")

lt = TOK // n
q64 = jnp.asarray(rng.standard_normal((lt, HEADS, NOPE + ROPE)), jnp.bfloat16)


@jax.jit
def tok(q, tk):
    kv_lens, phys = prepare_gather_indices(
        tk, pi, cu1, page_size=BS, total_num_pages=sp_pages)
    return sparse_ragged_paged_attention(
        q, sp_nope, sp_rope, phys[:lt], None, cu1, d1, sm_scale=0.06,
        prepared_kv_lens=kv_lens[:lt], prepared_token_end=jnp.int32(lt))


ms_tok = timeit(tok, q64, jnp.asarray(sp_topk))
print(f"token-sharded ({lt} tok x {HEADS} heads, incl. full-T page lookup): "
      f"{ms_tok:7.2f} ms")
