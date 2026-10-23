"""v3 indexer under speculative verify windows with partial acceptance.

    source vllm_env.sh; SRC=/zzlfs/glm53flash_torchtpu/vllm-prefix python test_indexer_spec_ring.py

Speculative run: after a prompt prefill, every step feeds each request a
window of K + 1 tokens at positions L..L+K (the last sampled token plus K
drafts), then accepts a random prefix of a tokens (1 <= a <= K + 1); rejected
positions carry garbage and are fed again (as real tokens) next step. The
member state is a ring of kpool + K rows inside a sparse-MLA nope pool, with
K + 1 checkpoint blocks per request and a random read offset each step, as
on the unified pool.

Reference run: the same real tokens, no speculation, ring = kpool, dense
member state, fed exactly the accepted chunks.

Every accepted position's top-k selection must match the reference bit for
bit, and so must the key record of every pool complete within the accepted
length.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(
    os.environ.get("SRC", "/zzlfs/glm53flash_torchtpu/vllm-prefix"), "src"))
from vllm_torchtpu.models.vllm.glm5_next.indexer import (  # noqa: E402
    _kpool_index_v3_jax, _with_pooled_member)

BS, HD, HEADS, KPOOL, K = 2048, 128, 32, 4, 3
RING = int(os.environ.get("RING", KPOOL + K))
SELECT_K = 512
WIDTH = -(-(SELECT_K * KPOOL + KPOOL - 1) // 256) * 256
MAX_LEN = 4096
SEQS = 4
PAGES = MAX_LEN // BS
NUM_KEY_BLOCKS = SEQS * PAGES + 2
NUM_POOL_BLOCKS = 1 + SEQS * (K + 1) + 3
ROWS_PER_PAGE = BS // KPOOL
STEPS = 60

rng = np.random.default_rng(0)
ape = jnp.asarray(rng.standard_normal((KPOOL, HD)) * 0.5, jnp.float32)
hot = rng.standard_normal(HD).astype(np.float32)


def true_token(s: int, p: int):
    """Deterministic real inputs for sequence s at position p."""
    r = np.random.default_rng(1000003 * s + p)
    k = r.standard_normal(HD).astype(np.float32)
    if r.random() < 0.05:
        k += 3.0 * hot
    return (k, r.standard_normal(HD).astype(np.float32),
            r.standard_normal((HEADS, HD)).astype(np.float32) + hot,
            r.uniform(0.0, 1.0, HEADS).astype(np.float32))


def garbage_token():
    return (rng.standard_normal(HD).astype(np.float32) * 5,
            rng.standard_normal(HD).astype(np.float32) * 5,
            rng.standard_normal((HEADS, HD)).astype(np.float32),
            rng.uniform(0.0, 1.0, HEADS).astype(np.float32))


def v3(c, kc, *a):
    return _kpool_index_v3_jax(c, kc, *a, kpool=KPOOL, select_k=SELECT_K,
                               width=WIDTH)


ref_op = jax.jit(v3, donate_argnums=(0, 1))
spec_op = jax.jit(_with_pooled_member(v3, RING), donate_argnums=(0, 1))

key_block_table = rng.permutation(NUM_KEY_BLOCKS)[:SEQS * PAGES].astype(np.int32)
ref_member = jnp.zeros((SEQS + 1, KPOOL, 2 * HD), jnp.bfloat16)
ref_slots = np.arange(1, SEQS + 1, dtype=np.int32)
ref_keys = jnp.zeros((NUM_KEY_BLOCKS, ROWS_PER_PAGE // 4, 4, 256), jnp.uint8)
pool = jnp.zeros((NUM_POOL_BLOCKS, BS, 4, 128), jnp.uint8)
ckpt = (1 + rng.permutation(NUM_POOL_BLOCKS - 1)[:SEQS * (K + 1)]).reshape(
    SEQS, K + 1).astype(np.int32)
state_blocks = ckpt[:, 0].copy()
spec_keys = jnp.zeros_like(ref_keys)


def pack(tokens_per_seq, done):
    """Batch rows from per-seq lists of token tuples starting at done[s]."""
    q_lens = [len(t) for t in tokens_per_seq]
    flat = [tok for t in tokens_per_seq for tok in t]
    t = len(flat)
    pad = max(8, -(-t // 8) * 8)
    k = np.zeros((pad, HD), np.float32)
    g = np.zeros((pad, HD), np.float32)
    q = np.zeros((pad, HEADS, HD), np.float32)
    w = np.zeros((pad, HEADS), np.float32)
    for i, (ki, gi, qi, wi) in enumerate(flat):
        k[i], g[i], q[i], w[i] = ki, gi, qi, wi
    seq_lens = np.asarray(done, np.int32) + np.asarray(q_lens, np.int32)
    cu = np.zeros(SEQS + 1, np.int32)
    cu[1:] = np.cumsum(q_lens)
    dist = np.array([0, 0, SEQS], np.int32)
    return t, (jnp.asarray(k, jnp.bfloat16), jnp.asarray(g, jnp.bfloat16),
               jnp.asarray(q), jnp.asarray(w), ape, jnp.asarray(seq_lens),
               jnp.asarray(cu), jnp.asarray(dist))


def run_ref(chunks, done):
    global ref_member, ref_keys
    t, args = pack(chunks, done)
    ref_member, ref_keys, out = ref_op(ref_member, ref_keys, *args,
                                       jnp.asarray(ref_slots),
                                       jnp.asarray(key_block_table))
    return np.asarray(out)[:t]


def run_spec(chunks, done):
    global pool, spec_keys
    t, args = pack(chunks, done)
    offsets = np.zeros(NUM_POOL_BLOCKS, np.int32)
    offsets[state_blocks] = rng.integers(0, K + 1, SEQS)
    pool, spec_keys, out = spec_op(pool, spec_keys, *args,
                                   jnp.asarray(state_blocks),
                                   jnp.asarray(key_block_table),
                                   jnp.asarray(ckpt), jnp.asarray(offsets))
    return np.asarray(out)[:t]


ok = True
prompt = [2101, 2333, 2050, 2600]  # past select_k * kpool, so selection matters
chunks = [[true_token(s, p) for p in range(prompt[s])] for s in range(SEQS)]
r, sp = run_ref(chunks, [0] * SEQS), run_spec(chunks, [0] * SEQS)
ok &= np.array_equal(r, sp)
print("prefill topk equal:", np.array_equal(r, sp))
done = list(prompt)
mismatch_steps = 0
for step in range(STEPS):
    accept = [int(a) for a in rng.integers(1, K + 2, SEQS)]
    windows = [[true_token(s, done[s] + j) if j < accept[s] else garbage_token()
                for j in range(K + 1)] for s in range(SEQS)]
    out = run_spec(windows, done)
    ref = run_ref([[true_token(s, done[s] + j) for j in range(accept[s])]
                   for s in range(SEQS)], done)
    # Rows of the accepted prefix of each window vs the reference chunk rows.
    so, ro = 0, 0
    for s in range(SEQS):
        if not np.array_equal(out[so:so + accept[s]], ref[ro:ro + accept[s]]):
            mismatch_steps += 1
            ok = False
        so += K + 1
        ro += accept[s]
    done = [d + a for d, a in zip(done, accept)]
print(f"{STEPS} verify steps, accepted lens now {done}: "
      f"steps with a mismatching accepted row: {mismatch_steps}")

rk, sk = np.asarray(ref_keys), np.asarray(spec_keys)
pages = key_block_table.reshape(SEQS, PAGES)
key_ok = True
for s in range(SEQS):
    for pool_idx in range(done[s] // KPOOL):
        row = pages[s, pool_idx // ROWS_PER_PAGE] * ROWS_PER_PAGE + pool_idx % ROWS_PER_PAGE
        blk, r4 = divmod(row, ROWS_PER_PAGE)
        if not np.array_equal(rk[blk, r4 // 4, r4 % 4], sk[blk, r4 // 4, r4 % 4]):
            key_ok = False
print("pool keys of complete accepted pools equal:", key_ok)
ok &= key_ok
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
