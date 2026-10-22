"""v3 indexer: member state on the unified pool vs the dense slot array.

    source vllm_env.sh; python test_indexer_pooled_member.py

Replays the same steps (chunked prefills whose boundaries land at every
offset within a pool, then decodes) through `_kpool_index_v3_jax` twice: once
with the dense ``[slots, kpool, 2 * hd]`` member state, once through
`_with_pooled_member` with the member rows inside a sparse-MLA nope pool
``(blocks, block_size, 4, 128)`` uint8 addressed by block id. Selections, the
pool-key cache and the member rows must match bit for bit, and the pooled run
must not touch any other pool row. The compiled pooled op must not copy the
pool (an XLA gather / scatter on the uint8 pool relayouts all of it).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(
    os.environ.get("SRC", "/zzlfs/glm53flash_torchtpu/vllm-torchtpu"), "src"))
from vllm_torchtpu.models.vllm.glm5_next.indexer import (  # noqa: E402
    _kpool_index_v3_jax, _with_pooled_member, gather_pooled_member)

BS, HD, HEADS, KPOOL = 2048, 128, 32, 4
SELECT_K = 512
WIDTH = -(-(SELECT_K * KPOOL + KPOOL - 1) // 256) * 256
MAX_LEN = 8192
MAX_SEQS = 4
NUM_KEY_BLOCKS = MAX_SEQS * (MAX_LEN // BS) + 4
NUM_POOL_BLOCKS = 16

rng = np.random.default_rng(0)
key_block_table = rng.permutation(NUM_KEY_BLOCKS)[: MAX_SEQS * (MAX_LEN // BS)].astype(np.int32)
ape = jnp.asarray(rng.standard_normal((KPOOL, HD)) * 0.5, jnp.float32)
hot = rng.standard_normal(HD).astype(np.float32)


def v3(c, kc, *a):
    return _kpool_index_v3_jax(c, kc, *a, kpool=KPOOL, select_k=SELECT_K,
                               width=WIDTH)


dense_op = jax.jit(v3, donate_argnums=(0, 1))
pooled_op = jax.jit(_with_pooled_member(v3, KPOOL), donate_argnums=(0, 1))

# Dense: slot 0 is the null slot; sequence s owns a shuffled slot.
dense = jnp.zeros((MAX_SEQS + 1, KPOOL, 2 * HD), jnp.bfloat16)
dense_slots = (rng.permutation(MAX_SEQS) + 1).astype(np.int32)
# Pooled: sequence s owns a shuffled state block; the rest of the pool is
# filled with a sentinel that must survive.
pool0 = rng.integers(0, 256, (NUM_POOL_BLOCKS, BS, 4, 128), dtype=np.uint8)
pool = jnp.asarray(pool0)
pool_blocks = (rng.permutation(NUM_POOL_BLOCKS - 1)[:MAX_SEQS] + 1).astype(np.int32)
# Both runs start from zeroed member rows (a fresh request's slot content is
# never read: see `_kpool_index_v3_jax`, but keep the comparison exact).
pool = pool.at[pool_blocks, :KPOOL].set(0)
key_dense = jnp.zeros((NUM_KEY_BLOCKS, BS // KPOOL // 4, 4, 256), jnp.uint8)
key_pooled = jnp.zeros_like(key_dense)


def step(q_lens, done, num_decode, pad_to):
    global dense, pool, key_dense, key_pooled
    t = int(sum(q_lens))
    k = rng.standard_normal((pad_to, HD)).astype(np.float32)
    k[:t] += 3.0 * hot * (rng.random((t, 1)) < 0.05)
    gate = rng.standard_normal((pad_to, HD)).astype(np.float32)
    q = rng.standard_normal((pad_to, HEADS, HD)).astype(np.float32) + hot
    w = rng.uniform(0.0, 1.0, (pad_to, HEADS)).astype(np.float32)
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[: len(q_lens)] = np.asarray(done) + np.asarray(q_lens)
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1: len(q_lens) + 1] = np.cumsum(q_lens)
    cu[len(q_lens) + 1:] = t
    dist = np.array([num_decode, num_decode, len(q_lens)], np.int32)
    args = (jnp.asarray(k, jnp.bfloat16), jnp.asarray(gate, jnp.bfloat16),
            jnp.asarray(q), jnp.asarray(w), ape, jnp.asarray(seq_lens),
            jnp.asarray(cu), jnp.asarray(dist))
    dense, key_dense, out_d = dense_op(dense, key_dense, *args,
                                       jnp.asarray(dense_slots),
                                       jnp.asarray(key_block_table))
    global last_args
    last_args = args + (jnp.asarray(pool_blocks), jnp.asarray(key_block_table))
    pool, key_pooled, out_p = pooled_op(pool, key_pooled, *last_args)
    return np.asarray(out_d)[:t], np.asarray(out_p)[:t]


def members_match() -> bool:
    d = np.asarray(dense)[dense_slots]
    p = np.asarray(gather_pooled_member(pool, jnp.asarray(pool_blocks), KPOOL)
                   ).reshape(MAX_SEQS, KPOOL, 2 * HD)
    return np.array_equal(d.view(np.uint16), p.view(np.uint16))


def pool_copies(args) -> list[str]:
    hlo = pooled_op.lower(pool, key_pooled, *args).compile().as_text()
    shape = "u8[%d,%d,4,128]" % (NUM_POOL_BLOCKS, BS)
    return [line.strip()[:160] for line in hlo.splitlines()
            if " copy(" in line and line.strip().split("=", 1)[-1].lstrip(
            ).startswith(shape)]


ok = True
targets = [7000, 3000, 1500, 5003]
done = [0, 0, 0, 0]
while any(d < tgt for d, tgt in zip(done, targets)):
    q_lens = [min(509, tgt - d) for d, tgt in zip(done, targets)]
    od, op = step(q_lens, done, 0, 2048)
    same = np.array_equal(od, op)
    ok &= same
    print(f"prefill done={done} topk_equal={same}")
    done = [d + ql for d, ql in zip(done, q_lens)]
for i in range(8):
    od, op = step([1] * 4, done, 4, 4)
    same = np.array_equal(od, op)
    ok &= same
    print(f"decode {i} topk_equal={same}")
    done = [d + 1 for d in done]

keys_same = np.array_equal(np.asarray(key_dense), np.asarray(key_pooled))
mem_same = members_match()
untouched = np.asarray(pool).copy()
untouched[pool_blocks, :KPOOL] = pool0[pool_blocks, :KPOOL]
others_same = np.array_equal(untouched, pool0)
print(f"pool-key cache equal={keys_same} member rows equal={mem_same} "
      f"other pool rows untouched={others_same}")
copies = pool_copies(last_args)
print(f"full-pool copies in the compiled pooled op: {len(copies)}")
for line in copies:
    print("   ", line)
ok &= keys_same and mem_same and others_same and not copies
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
