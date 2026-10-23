#!/usr/bin/env python
"""Single-stream decode speed, no streaming and no logprobs (both slow the
server down enough to hide a speculative-decoding speedup).

    python decode_speed.py --url ref=http://10.128.0.183:8000 --url mtp=http://10.128.0.184:8000

decode tok/s = 255 / (t(256 tokens) - t(1 token)), median of --reps fresh
prompts per length (fresh so the prefix cache cannot help either request).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ab import prompt_ids  # noqa: E402


def timed(url: str, model: str, ids: list[int], n: int) -> float:
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": n,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    urllib.request.urlopen(req, timeout=1800).read()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True, help="name=url")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--tokenizer",
                    default="/home/zzl_google_com/models/GLM-5.3-Flash/tokenizer.json")
    ap.add_argument("--lengths", default="300,1000,4000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    servers = [u.split("=", 1) for u in args.url]
    for n in (int(x) for x in args.lengths.split(",")):
        res = {}
        for k, (name, url) in enumerate(servers):
            v = []
            for r in range(args.reps):
                seed = args.seed * 1000 + n * 10 + r * 100 + k
                t1 = timed(url, args.model, prompt_ids(tok, n, seed), 1)
                t256 = timed(url, args.model, prompt_ids(tok, n, seed + 50), 256)
                v.append(255 / (t256 - t1))
            res[name] = statistics.median(v)
        print(f"prompt {n:6d}: " + " | ".join(
            f"{name} {tps:.0f} tok/s" for name, tps in res.items()), flush=True)


if __name__ == "__main__":
    main()
