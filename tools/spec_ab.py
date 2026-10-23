#!/usr/bin/env python
"""Speculative decoding A/B: the same greedy request against a server with MTP
and one without; the generated tokens must match.

    python spec_ab.py --spec http://10.128.0.184:8000 --ref http://10.128.0.183:8000

Prompts span both sides of 2048 tokens: below it the DSA indexer selects
every pool, so a broken indexer state cannot change the output; above it, it
can. Also reports the MTP acceptance (vllm:spec_decode_* counters) and the
per-request decode speed of both servers.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request

from tokenizers import Tokenizer

WORDS = ("ledger pallet aisle scan truck inspector warehouse inbound outbound "
         "crate invoice forklift dock shelf barcode shipment").split()


def prompt_ids(tok: Tokenizer, n: int, seed: int) -> list[int]:
    rnd = random.Random(seed)
    head = (f"Below is warehouse log {seed}. Read it, then write a detailed "
            f"summary of which aisles and pallets appear most often.\n")
    ids = tok.encode(head).ids
    line = 0
    while len(ids) < n:
        words = " ".join(rnd.choice(WORDS) for _ in range(12))
        ids += tok.encode(f"line {line}: {words}\n").ids
        line += 1
    tail = tok.encode("\nSummary:").ids
    return ids[: n - len(tail)] + tail


def stream(url: str, model: str, ids: list[int], max_tokens: int):
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True, "logprobs": 1,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    tokens: list[str] = []
    lps: list[float] = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[5:])
            for c in ev.get("choices") or []:
                lp = c.get("logprobs") or {}
                if lp.get("tokens"):
                    if first is None:
                        first = time.perf_counter() - t0
                    tokens += lp["tokens"]
                    lps += lp["token_logprobs"]
    total = time.perf_counter() - t0
    if first is None:
        raise RuntimeError(f"{url} returned no tokens")
    decode_tps = (len(tokens) - 1) / (total - first) if len(tokens) > 1 else 0.0
    return tokens, lps, first, decode_tps


def spec_counters(url: str) -> tuple[float, float]:
    text = urllib.request.urlopen(f"{url}/metrics", timeout=60).read().decode()

    def total(name):
        return sum(float(v) for v in re.findall(
            rf"^vllm:{name}_total\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M))

    return total("spec_decode_num_accepted_tokens"), total(
        "spec_decode_num_draft_tokens")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--tokenizer",
                    default="/home/zzl_google_com/models/GLM-5.3-Flash/tokenizer.json")
    ap.add_argument("--lengths", default="100,1000,3000,8000,14000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    ok = True
    for i, n in enumerate(int(x) for x in args.lengths.split(",")):
        ids = prompt_ids(tok, n, args.seed * 100 + i)
        a0, d0 = spec_counters(args.spec)
        st, sl, sttft, stps = stream(args.spec, args.model, ids, args.max_tokens)
        a1, d1 = spec_counters(args.spec)
        rt, rl, rttft, rtps = stream(args.ref, args.model, ids, args.max_tokens)
        div = next((j for j, (x, y) in enumerate(zip(st, rt)) if x != y), None)
        same = st == rt
        ok &= same
        acc = (a1 - a0) / (d1 - d0) if d1 > d0 else float("nan")
        msg = (f"len {len(ids):6d}: tokens equal {same}"
               f"{'' if same else f' (first diff at {div} of {len(rt)})'} | "
               f"acceptance {acc:.2f} | decode tok/s spec {stps:.0f} ref {rtps:.0f}"
               f" | ttft spec {sttft:.2f}s ref {rttft:.2f}s")
        if not same and div is not None:
            lo = max(0, div - 3)
            msg += (f"\n   spec: {''.join(st[lo:div + 5])!r} lp {sl[div]:.3f}"
                    f"\n   ref : {''.join(rt[lo:div + 5])!r} lp {rl[div]:.3f}")
        print(msg, flush=True)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
