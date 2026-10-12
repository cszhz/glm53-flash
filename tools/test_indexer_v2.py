"""v3 indexer (per-request member state + fp8 pool-key cache + streamindex_topk)
against v1 (paged members, XLA scoring), on TPU.

    source vllm_env.sh; python test_indexer_v2.py

Replays the same steps through both ops -- chunked prefills, then decodes,
several sequences at once -- each op keeping its own caches. Rows whose
visible pool count fits in `select_k` select every pool, so there the two
must agree exactly; past that, fp8 keys may reorder near-ties, so report the
overlap of the selected pool sets. Every row must also be left-packed.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/zzlfs/glm53flash_torchtpu/vllm-torchtpu/src")
from vllm_torchtpu.models.vllm.glm5_next.indexer import (  # noqa: E402
    _kpool_index_jax, _kpool_index_v3_jax)

BS, HD, HEADS, KPOOL = 256, 128, 32, 4
SELECT_K = 512
WIDTH = -(-(SELECT_K * KPOOL + KPOOL - 1) // 256) * 256
MAX_LEN = 8192
PAGES_PER_SEQ = MAX_LEN // BS
MAX_SEQS = 4
NUM_BLOCKS = MAX_SEQS * PAGES_PER_SEQ + 8

rng = np.random.default_rng(0)
block_table = rng.permutation(NUM_BLOCKS)[: MAX_SEQS * PAGES_PER_SEQ].astype(np.int32)
# The key cache is its own KV cache group, so it gets its own block table.
key_block_table = rng.permutation(NUM_BLOCKS)[: MAX_SEQS * PAGES_PER_SEQ].astype(np.int32)
ape = jnp.asarray(rng.standard_normal((KPOOL, HD)) * 0.5, jnp.float32)
# A fixed key direction shared by every sequence, so scores have structure
# (a few clearly-best pools) rather than being pure noise ties.
hot = rng.standard_normal(HD).astype(np.float32)

v1 = jax.jit(lambda c, *a: _kpool_index_jax(
    c, *a, kpool=KPOOL, select_k=SELECT_K, width=WIDTH,
    max_pools=MAX_LEN // KPOOL, num_heads=HEADS), donate_argnums=(0,))
v2 = jax.jit(lambda c, kc, *a: _kpool_index_v3_jax(
    c, kc, *a, kpool=KPOOL, select_k=SELECT_K, width=WIDTH), donate_argnums=(0, 1))

cache1 = jnp.zeros((NUM_BLOCKS, BS // 2, 2, 2 * HD), jnp.bfloat16)
# Slot 0 is the null slot; sequence s owns slot s + 1 (shuffled below so a
# slot/sequence mix-up cannot pass).
cache2 = jnp.zeros((MAX_SEQS + 1, KPOOL, 2 * HD), jnp.bfloat16)
state_indices = (rng.permutation(MAX_SEQS) + 1).astype(np.int32)
key_cache = jnp.zeros((NUM_BLOCKS, BS // KPOOL // 4, 4, 256), jnp.uint8)


def step(q_lens, done, num_decode, pad_to):
    """One engine step: sequence s gets q_lens[s] new tokens after done[s]."""
    global cache1, cache2, key_cache
    t = int(sum(q_lens))
    k = rng.standard_normal((pad_to, HD)).astype(np.float32)
    k[:t] += 3.0 * hot * (rng.random((t, 1)) < 0.05)  # a few "relevant" tokens
    gate = rng.standard_normal((pad_to, HD)).astype(np.float32)
    q = rng.standard_normal((pad_to, HEADS, HD)).astype(np.float32) + hot
    w = rng.uniform(0.0, 1.0, (pad_to, HEADS)).astype(np.float32)
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[: len(q_lens)] = np.asarray(done) + np.asarray(q_lens)
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : len(q_lens) + 1] = np.cumsum(q_lens)
    cu[len(q_lens) + 1 :] = t
    dist = np.array([num_decode, num_decode, len(q_lens)], np.int32)
    head = (jnp.asarray(k, jnp.bfloat16), jnp.asarray(gate, jnp.bfloat16),
            jnp.asarray(q), jnp.asarray(w), ape, jnp.asarray(seq_lens))
    cache1, out1 = v1(cache1, *head, jnp.asarray(block_table), jnp.asarray(cu))
    cache2, key_cache, out2 = v2(cache2, key_cache, *head, jnp.asarray(cu),
                                 jnp.asarray(dist), jnp.asarray(state_indices),
                                 jnp.asarray(key_block_table))
    return np.asarray(out1)[:t], np.asarray(out2)[:t], seq_lens, cu


def positions(seq_lens, cu, t):
    tok = np.arange(t)
    sid = np.searchsorted(cu[1:], tok, side="right")
    return seq_lens[sid] - (cu[sid + 1] - cu[sid]) + (tok - cu[sid])


def check(out1, out2, label, pos):
    exact_rows = mismatched_exact = 0
    overlaps = []
    for r in range(out1.shape[0]):
        a, b = out1[r], out2[r]
        for name, row in (("v1", a), ("v2", b)):
            n = int((row >= 0).sum())
            assert (row[:n] >= 0).all() and (row[n:] < 0).all(), \
                f"{label} row {r}: {name} not left-packed"
        pa, pb = set(a[a >= 0].tolist()), set(b[b >= 0].tolist())
        if (pos[r] + 1) // KPOOL <= SELECT_K:  # every visible pool was selected
            exact_rows += 1
            mismatched_exact += pa != pb
        else:
            overlaps.append(len(pa & pb) / len(pa))
    msg = f"{label:34s} exact rows {exact_rows:5d} (mismatch {mismatched_exact})"
    if overlaps:
        msg += f" | top-k rows {len(overlaps):5d}, overlap min {min(overlaps):.4f} mean {np.mean(overlaps):.4f}"
    print(msg)
    return mismatched_exact == 0 and (not overlaps or min(overlaps) > 0.97)


ok = True
# Sequence lengths straddle the 2048-token (512-pool) selection width.
targets = [7000, 3000, 1500, 5003]
done = [0, 0, 0, 0]
CHUNK = 2048
while any(d < tgt for d, tgt in zip(done, targets)):
    # 509 per sequence: chunk boundaries land at every offset within a pool.
    q_lens = [min(509, tgt - d) for d, tgt in zip(done, targets)]
    o1, o2, sl, cu = step(q_lens, done, 0, CHUNK)
    ok &= check(o1, o2, f"prefill chunk, done={done}", positions(sl, cu, len(o1)))
    done = [d + ql for d, ql in zip(done, q_lens)]
for i in range(12):
    o1, o2, sl, cu = step([1] * 4, done, 4, 4)
    ok &= check(o1, o2, f"decode step {i}, lens={[d + 1 for d in done]}",
                positions(sl, cu, len(o1)))
    done = [d + 1 for d in done]
print("PASS" if ok else "FAIL")
