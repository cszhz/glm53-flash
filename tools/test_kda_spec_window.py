"""Pooled KDA on GLM-5.3's nope pool under speculative verify windows.

    source vllm_env.sh; SRC=... python test_kda_spec_window.py

After a 300-token prefill, every step feeds a window of K + 1 tokens (the
true next token plus K drafts), accepts a random prefix, and the next step
resumes from checkpoint (accepted - 1) -- what the runner does. Accepted
tokens' outputs must match a non-speculative run that only ever saw the
true tokens, one per step.
"""
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(
    os.environ.get("SRC", "/zzlfs/glm53flash_torchtpu/vllm-prefix"), "src"))
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (  # noqa: E402
    _pooled_kda_core)

H, D, KC = 8, 128, 4
NB, BS = 8, 2048
K = int(os.environ.get("K", "1"))
LOWER, EPS = -5.0, 1e-6
rng = np.random.default_rng(0)


def bf16(shape, scale=0.5, r=rng):
    return jnp.asarray(r.standard_normal(shape).astype(np.float32) * scale, jnp.bfloat16)


W = dict(conv_weight=bf16((KC, 3, H, D)),
         a_log=jnp.asarray(rng.standard_normal(H).astype(np.float32), jnp.float32),
         dt_bias=jnp.asarray(rng.standard_normal(H * D).astype(np.float32), jnp.float32),
         norm_weight=bf16((D,)))


def token(p, true=True):
    r = np.random.default_rng(100003 + p if true else int(rng.integers(1 << 30)))
    return (bf16((1, 3 * H * D), r=r), bf16((1, H * D), r=r), bf16((1, H), r=r), bf16((1, H * D), r=r))


def cat(toks):
    return [jnp.concatenate([t[i] for t in toks]) for i in range(4)]


def call(pool, toks, seq_len, kind, spec, offsets=None):
    q = len(toks)
    x = cat(toks)
    dist = {"prefill": [0, 0, 1], "decode": [1, 1, 1], "verify": [0, 0, 1]}[kind]
    wdist = [1, 1, 1] if kind in ("decode", "verify") else [0, 0, 1]
    extra = {}
    if spec:
        extra = dict(window_distribution=jnp.asarray(wdist, jnp.int32),
                     slot_read_offsets=offsets,
                     ckpt_indices=jnp.asarray([list(range(3, 3 + K + 1))], jnp.int32))
    return _pooled_kda_core(*x, pool, W["conv_weight"], W["a_log"], W["dt_bias"],
                            W["norm_weight"], jnp.asarray([0, q], jnp.int32),
                            jnp.asarray([3], jnp.int32), jnp.asarray([seq_len], jnp.int32),
                            jnp.asarray(dist, jnp.int32), lower_bound=LOWER, eps=EPS,
                            pool_block_tokens=BS, num_spec_tokens=K if spec else 0,
                            pool_tokens_per_row=1, **extra)


P0 = 300
ref_pool = jnp.zeros((NB, BS, 4, 128), jnp.uint8)
spec_pool = jnp.zeros((NB, BS, 4, 128), jnp.uint8)
pre = [token(p) for p in range(P0)]
ref_out, ref_pool = call(ref_pool, pre, P0, "prefill", False)
offsets = jnp.zeros(NB, jnp.int32)
spec_out, spec_pool = call(spec_pool, pre, P0, "prefill", True, offsets)
print("prefill equal:", bool(np.array_equal(np.asarray(ref_out), np.asarray(spec_out))))

# Reference: one true token per decode step.
ref_tok = {}
pos = P0
for p in range(P0, P0 + 80):
    o, ref_pool = call(ref_pool, [token(p)], p + 1, "decode", False)
    ref_tok[p] = np.asarray(o[0], np.float32)

bad, worst, steps = 0, 0.0, 0
pos = P0
while pos < P0 + 70:
    a = int(rng.integers(1, K + 2))
    toks = [token(pos)] + [token(pos + j) if j < a else token(pos + j, False) for j in range(1, K + 1)]
    out, spec_pool = call(spec_pool, toks, pos + K + 1, "verify", True, offsets)
    for j in range(a):
        d = np.abs(np.asarray(out[j], np.float32) - ref_tok[pos + j]).max() / max(np.abs(ref_tok[pos + j]).max(), 1e-6)
        worst = max(worst, d)
        bad += d > 2e-2
    offsets = offsets.at[3].set(a - 1)
    pos += a
    steps += 1
print(f"K={K}: {steps} verify steps, accepted tokens {pos - P0}, rows off by >2% {bad}, worst rel diff {worst:.2e}")
print("PASS" if bad == 0 else "FAIL")
sys.exit(0 if bad == 0 else 1)
