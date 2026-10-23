"""Pooled KDA on GLM-5.3's nope pool: speculative mode must not change prefill.

    source vllm_env.sh; SRC=/zzlfs/glm53flash_torchtpu/vllm-prefix python test_kda_spec_vs_nospec.py

GLM-5.3 TP8 shard geometry (8 heads, head_dim 128, conv 4, gate lower bound
-5) on the sparse-MLA nope cache (blocks, 2048, 4, 128) uint8 with
``pool_tokens_per_row=1``. One fresh request runs a prefill chunk, a
continuation chunk and three decode steps, once with ``num_spec_tokens=0``
(state in its block) and once with ``num_spec_tokens=1`` (two checkpoint
blocks, read offset 0, which is what a spec-enabled server does for a
request that has not verified anything yet). Outputs must agree.
"""
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(
    os.environ.get("SRC", "/zzlfs/glm53flash_torchtpu/vllm-prefix"), "src"))
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (  # noqa: E402
    _pooled_kda_core)

H, D, K = 8, 128, 4
NB, BS = 8, 2048
LOWER, EPS = -5.0, 1e-6

rng = np.random.default_rng(0)


def bf16(shape, scale=0.5):
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * scale,
                       jnp.bfloat16)


weights = dict(
    conv_weight=bf16((K, 3, H, D)),
    a_log=jnp.asarray(rng.standard_normal(H).astype(np.float32), jnp.float32),
    dt_bias=jnp.asarray(rng.standard_normal(H * D).astype(np.float32), jnp.float32),
    norm_weight=bf16((D,)),
)


def step_inputs(q_len):
    return dict(mixed_qkv=bf16((q_len, 3 * H * D)), raw_gate=bf16((q_len, H * D)),
                beta=bf16((q_len, H)), output_gate=bf16((q_len, H * D)))


PAD = int(os.environ.get("PAD_REQS", "8"))       # batch padded like the runner's
CKPT1 = int(os.environ.get("CKPT1", "0"))        # 0: second checkpoint column off-row (null block)


def run(pool, inp, q_len, seq_len, decode, spec):
    qsl = jnp.asarray([0] + [q_len] * PAD, jnp.int32)
    sl = jnp.asarray([seq_len] + [0] * (PAD - 1), jnp.int32)
    dist = jnp.asarray([1, 1, PAD] if decode else [0, 0, PAD], jnp.int32)
    state = jnp.asarray([3] + [0] * (PAD - 1), jnp.int32)
    extra = {}
    if spec:
        ckpt = np.zeros((PAD, 2), np.int32)
        ckpt[0] = [3, CKPT1]
        extra = dict(window_distribution=dist,
                     slot_read_offsets=jnp.zeros(NB, jnp.int32),
                     ckpt_indices=jnp.asarray(ckpt))
    out, pool = _pooled_kda_core(
        inp["mixed_qkv"], inp["raw_gate"], inp["beta"], inp["output_gate"], pool,
        weights["conv_weight"], weights["a_log"], weights["dt_bias"],
        weights["norm_weight"], qsl, state, sl, dist,
        lower_bound=LOWER, eps=EPS, pool_block_tokens=BS,
        num_spec_tokens=1 if spec else 0, pool_tokens_per_row=1, **extra)
    return out, pool


ok = True
pools = {False: jnp.zeros((NB, BS, 4, 128), jnp.uint8),
         True: jnp.zeros((NB, BS, 4, 128), jnp.uint8)}
seq = 0
for name, q_len, decode in (("prefill 300", 300, False), ("continue 200", 200, False),
                            ("decode", 1, True), ("decode", 1, True), ("decode", 1, True)):
    seq += q_len
    inp = step_inputs(q_len)
    outs = {}
    for spec in (False, True):
        outs[spec], pools[spec] = run(pools[spec], inp, q_len, seq, decode, spec)
    a = np.asarray(outs[False], np.float32)
    b = np.asarray(outs[True], np.float32)
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-6)
    same = np.array_equal(a, b)
    ok &= rel < 2e-2
    print(f"{name:13s} seq {seq:4d}: bitwise {same}  max rel diff {rel:.2e}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
