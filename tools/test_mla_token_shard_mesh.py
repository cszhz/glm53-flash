"""Token-sharded sparse MLA on a real 8-device mesh (pure JAX, no torch).

    source vllm_env.sh; SRC=<worktree> python test_mla_token_shard_mesh.py

One request prefilled from position 0 with GLM-shaped selections (every
visible token while pos < 2051, left-packed, position order), padded to the
compile bucket. `sparse_mla_attention_token_sharded` over the 8-device mesh
(heads sharded in, all-to-all, token-sharded kernel, all-to-all back) against
`sparse_mla_attention` with all 64 heads on one device.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

sys.path.insert(0, os.environ.get(
    "SRC", "/zzlfs/glm53flash_torchtpu/vllm-torchtpu") + "/src")
from vllm_torchtpu.kernels.mla.kv_cache_utils import (  # noqa: E402
    KVCacheLayout, KVCacheType, SparseMLAKVCacheSpec)
from vllm_torchtpu.layers.core.attention_interface import (  # noqa: E402
    sparse_mla_attention, sparse_mla_attention_local_tokens,
    sparse_mla_attention_token_sharded)

HEADS, NOPE, ROPE, WIDTH, BS = 64, 512, 64, 2304, 256
devs = jax.devices()
mesh8 = Mesh(np.asarray(devs[:8]), ("x", ))
mesh1 = Mesh(np.asarray(devs[:1]), ("model", ))
rng = np.random.default_rng(0)


def run(T, real, ctx0=0):
    pages = -(-(ctx0 + T) // BS) + 2
    nspec = SparseMLAKVCacheSpec.create(KVCacheType.NOPE, KVCacheLayout.TENSORCORE,
                                        pages, BS, NOPE, 4)
    rspec = SparseMLAKVCacheSpec.create(KVCacheType.ROPE, KVCacheLayout.TENSORCORE,
                                        pages, BS, ROPE, 4)
    nope = jnp.asarray(rng.integers(0, 120, (pages, BS, 4, 128)), jnp.uint8)
    rope = jnp.zeros((pages, BS // 4, 4, 128), jnp.uint8)
    topk = np.full((T, WIDTH), -1, np.int32)
    for r in range(real):
        vis = ctx0 + r + 1
        if vis <= 2051:
            topk[r, :vis] = np.arange(vis)
        else:
            topk[r, :2051] = np.sort(rng.choice(vis, 2051, replace=False))
    ql = jnp.asarray(rng.standard_normal((T, HEADS, NOPE)), jnp.bfloat16)
    qpe = jnp.zeros((T, HEADS, ROPE), jnp.bfloat16)
    kv = jnp.asarray(rng.standard_normal((T, NOPE)), jnp.float8_e4m3fn)
    kpe = jnp.zeros((T, ROPE), jnp.float8_e4m3fn)
    meta = (jnp.asarray(topk), jnp.asarray([ctx0 + real], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32), jnp.asarray([0, real], jnp.int32),
            jnp.asarray([0, 0, 1], jnp.int32))

    _, _, ref = sparse_mla_attention(ql, qpe, kv, kpe, nope, rope, *meta, mesh1,
                                     nspec, rspec, sm_scale=0.06, k_scale=1.0)
    heads = NamedSharding(mesh8, P(None, "x", None))
    rep = NamedSharding(mesh8, P())
    put = lambda x, s: jax.device_put(x, s)  # noqa: E731
    _, _, got = sparse_mla_attention_token_sharded(
        put(ql, heads), put(qpe, heads), put(kv, rep), put(kpe, rep),
        put(nope, rep), put(rope, rep), *[put(m, rep) for m in meta], mesh8,
        nspec, rspec, sm_scale=0.06, k_scale=1.0, axis_name="x", min_tokens=128)
    report("token-sharded", T, real, ctx0, ref, got)

    # SP: each device holds its own T/8 rows of q / topk with all heads. P()
    # specs, different data per device -- what the torch side hands the op.
    lt = T // 8

    def per_device(x):
        x = np.asarray(x)
        return jax.make_array_from_single_device_arrays(
            (lt, ) + x.shape[1:], rep,
            [jax.device_put(x[i * lt:(i + 1) * lt], d)
             for i, d in enumerate(mesh8.devices.flat)])

    _, _, got_sp = sparse_mla_attention_local_tokens(
        per_device(ql), per_device(qpe), put(kv, rep), put(kpe, rep),
        put(nope, rep), put(rope, rep), per_device(topk),
        *[put(m, rep) for m in meta[1:]], mesh8, nspec, rspec, sm_scale=0.06,
        k_scale=1.0, axis_name="x")
    got_sp = np.concatenate([np.asarray(s.data) for s in sorted(
        got_sp.addressable_shards, key=lambda s: s.device.id)])
    report("local-tokens", T, real, ctx0, ref, got_sp)


def report(name, T, real, ctx0, ref, got):
    d = np.abs(np.asarray(ref, np.float32)[:real] -
               np.asarray(got, np.float32)[:real]).max(axis=(1, 2))
    bad = np.nonzero(d > 0)[0]
    print(f"{name:13s} T={T} real={real} ctx0={ctx0}: max|diff| {d.max():.3g}, "
          f"rows differing {len(bad)}" + (f" (e.g. {bad[:8].tolist()})" if len(bad) else ""))


for T, real, ctx0 in ((512, 240, 0), (1024, 540, 0), (2048, 1550, 0),
                      (4096, 3050, 0), (8192, 7550, 0), (1024, 800, 9000)):
    run(T, real, ctx0)
