#!/usr/bin/env python
"""Single-stream TTFT / decode-rate probe against an OpenAI-compatible server.

Deliberately minimal: one request at a time, streaming, so the number is the
model's own latency and not a scheduler artefact.

    python ttft_bench.py --url http://10.128.0.127:8000 --tokens 1000,4000,7000
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from needle_test import build_prompt


def one(url: str, model: str, prompt: str, max_tokens: int) -> tuple[float,
                                                                     float,
                                                                     int]:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if not chunk["choices"][0].get("text"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - start
            n += 1
    return ttft, time.perf_counter() - start, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--fillers", default="20,60,120,200,300")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    print(f"{'prompt tok':>10} {'TTFT s':>8} {'decode tok/s':>13} "
          f"{'total s':>8}")
    for filler in (int(f) for f in args.fillers.split(",")):
        prompt = build_prompt(filler, 0.5)
        one(args.url, args.model, prompt, 1)  # warm the bucket
        ttft, total, n = one(args.url, args.model, prompt, args.max_tokens)
        rate = (n - 1) / (total - ttft) if n > 1 else float("nan")
        # Token count is the server's; approximate from the prompt for the
        # label only (chars/4 is close enough for a monotone x axis).
        print(f"{len(prompt) // 4:>10} {ttft:>8.3f} {rate:>13.1f} "
              f"{total:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
