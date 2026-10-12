#!/usr/bin/env python
"""Needle-in-a-haystack probe for the DSA indexer.

The point is not retrieval quality, it is whether the k-pool indexer selects
the right pools: the needle sits in the middle of a prompt far longer than
`index_topk` (2048), so a broken indexer can only answer by luck. Run the same
prompt against a dense build (`GLM53_DENSE_DSA=1`) and a sparse one and diff.

    /zzlfs/glm53flash_torchtpu/.venv/bin/python /zzlfs/glm53flash_torchtpu/needle_test.py \
        --url http://10.128.0.127:8000 --filler 600
"""
from __future__ import annotations

import argparse
import json
import urllib.request

NEEDLE_CODE = "QX-7391-ZULU"


def build_prompt(num_filler: int, depth: float) -> str:
    lines = [
        f"Record {i:04d}: the maintenance window for cluster node {i} closes "
        f"at {(i * 7) % 24:02d}:{(i * 13) % 60:02d} UTC."
        for i in range(num_filler)
    ]
    needle = (f"Record {int(num_filler * depth):04d}-SPECIAL: the emergency "
              f"access code for the primary datacenter is {NEEDLE_CODE}. "
              f"Remember it.")
    lines.insert(int(num_filler * depth), needle)
    return ("Below is a log file.\n\n" + "\n".join(lines) +
            "\n\nQuestion: what is the emergency access code for the primary "
            "datacenter? Answer with the code only.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--filler", type=int, default=600)
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    prompt = build_prompt(args.filler, args.depth)
    body = json.dumps({
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }).encode()
    req = urllib.request.Request(f"{args.url}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.load(resp)

    usage = out.get("usage", {})
    message = out["choices"][0]["message"]
    text = (message.get("content") or "")
    reasoning = (message.get("reasoning_content") or "")
    print(f"prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')}")
    if reasoning:
        print(f"--- reasoning ---\n{reasoning.strip()[:800]}")
    print(f"--- answer ---\n{text.strip()[:800]}")
    found = NEEDLE_CODE in text or NEEDLE_CODE in reasoning
    print(f"\nneedle {'FOUND' if found else 'MISSING'} ({NEEDLE_CODE})")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
