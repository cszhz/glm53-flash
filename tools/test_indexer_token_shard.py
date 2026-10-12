"""Token-sharded indexer selection vs the whole-batch selection, on one TPU.

    source vllm_env.sh; SRC=<worktree> python test_indexer_token_shard.py

Replays chunked prefills and mixed decode+prefill steps through the v3
indexer, then re-selects each step's rows as `--shards` token slices with
`_select_pools_slice` (what each rank runs in a token-sharded step) against
the same updated key cache. Real rows must match exactly.
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.environ.get(
    "SRC", "/zzlfs/glm53flash_torchtpu/vllm-torchtpu") + "/src")
from vllm_torchtpu.models.vllm.glm5_next.indexer import (  # noqa: E402
    _kpool_index_v3_jax, _select_pools_slice, _step_positions)

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=8)
a = ap.parse_args()

BS, HD, HEADS, KPOOL = 256, 128, 32, 4
SELECT_K = 512
WIDTH = -(-(SELECT_K * KPOOL + KPOOL - 1) // 256) * 256
MAX_LEN = 16384
PAGES_PER_SEQ = MAX_LEN // BS
MAX_SEQS = 6
NUM_BLOCKS = MAX_SEQS * PAGES_PER_SEQ + 8
KW = dict(kpool=KPOOL, select_k=SELECT_K, width=WIDTH)

rng = np.random.default_rng(0)
key_block_table = rng.permutation(NUM_BLOCKS)[: MAX_SEQS * PAGES_PER_SEQ].astype(np.int32)
ape = jnp.asarray(rng.standard_normal((KPOOL, HD)) * 0.5, jnp.float32)
hot = rng.standard_normal(HD).astype(np.float32)
state = jnp.zeros((MAX_SEQS + 1, KPOOL, 2 * HD), jnp.bfloat16)
state_indices = (rng.permutation(MAX_SEQS) + 1).astype(np.int32)
key_cache = jnp.zeros((NUM_BLOCKS, BS // KPOOL // 4, 4, 256), jnp.uint8)

full = jax.jit(lambda c, kc, *args: _kpool_index_v3_jax(c, kc, *args, **KW),
               donate_argnums=(0, 1))


@jax.jit
def sharded(key_cache, q, w, seq_lens, cu, dist, pages):
    t = q.shape[0]
    seq_id, pos, live = _step_positions(seq_lens, cu, t)
    lt = t // a.shards
    return jnp.concatenate([
        _select_pools_slice(q, w, key_cache, seq_lens, cu, dist, pages, pos,
                            live, jnp.int32(s * lt), lt, **KW)
        for s in range(a.shards)])


def step(q_lens, done, num_decode, pad_to):
    global state, key_cache
    t = int(sum(q_lens))
    k = rng.standard_normal((pad_to, HD)).astype(np.float32)
    k[:t] += 3.0 * hot * (rng.random((t, 1)) < 0.05)
    gate = rng.standard_normal((pad_to, HD)).astype(np.float32)
    q = jnp.asarray(rng.standard_normal((pad_to, HEADS, HD)).astype(np.float32) + hot)
    w = jnp.asarray(rng.uniform(0.0, 1.0, (pad_to, HEADS)).astype(np.float32))
    n = len(q_lens)
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:n] = np.asarray(done) + np.asarray(q_lens)
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1:n + 1] = np.cumsum(q_lens)
    cu[n + 1:] = t
    dist = jnp.asarray([num_decode, num_decode, n], jnp.int32)
    seq_lens, cu = jnp.asarray(seq_lens), jnp.asarray(cu)
    pages = jnp.asarray(key_block_table)
    state, key_cache, ref = full(
        state, key_cache, jnp.asarray(k, jnp.bfloat16),
        jnp.asarray(gate, jnp.bfloat16), q, w, ape, seq_lens, cu, dist,
        jnp.asarray(state_indices), pages)
    got = sharded(key_cache, q, w, seq_lens, cu, dist, pages)
    ref, got = np.asarray(ref)[:t], np.asarray(got)[:t]
    bad = int((ref != got).any(axis=1).sum())
    print(f"q_lens={q_lens} decode={num_decode} pad={pad_to}: "
          f"{t} rows, {bad} differ")
    return bad == 0


ok = True
targets = [12000, 3000, 9001, 600]
done = [0] * 4
while any(d < tgt for d, tgt in zip(done, targets)):
    q_lens = [min(1531, tgt - d) for d, tgt in zip(done, targets)]
    ok &= step(q_lens, done, 0, 8192)
    done = [d + ql for d, ql in zip(done, q_lens)]
# Decodes in front of two new prefills: decode requests split across slices.
for i in range(3):
    q_lens = [1, 1, 1, 1, 700, 300]
    ok &= step(q_lens, done + [i * 700, i * 300], 4, 1024)
    done = [d + 1 for d in done]
print("PASS" if ok else "FAIL")
