#!/usr/bin/env python
"""Record / compare next-token logprobs across server configurations.

Free-running greedy text is a terrible A/B: one 1e-3 logit difference early in
a chain of thought rewrites the whole continuation. This asks for a *single*
token with `logprobs`, so what comes back is a direct read of the model's
distribution at one point.

Dumping to a file decouples the runs, which matters because a config change
(fp8 vs bf16 KV, dense vs sparse DSA) costs a 25-minute restart, so the two
sides of a comparison rarely exist at the same time:

    python logprob_ab.py --url http://10.128.0.127:8000 --dump ab/sparse_fp8.json
    python logprob_ab.py --compare ab/dense_fp8.json ab/sparse_fp8.json
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from needle_test import build_prompt

FILLERS = (1, 3, 8, 20, 60, 120, 200, 300)


def top_logprobs(url: str, model: str, prompt: str, n: int) -> dict[str, float]:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": n,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.load(resp)
    return out["choices"][0]["logprobs"]["top_logprobs"][0]


def compare(a: dict, b: dict, name_a: str, name_b: str) -> None:
    print(f"{name_a}  vs  {name_b}")
    worst = 0.0
    agree = 0
    keys = [k for k in a if k in b]
    for key in keys:
        pa, pb = a[key], b[key]
        top_a = max(pa, key=pa.get)
        top_b = max(pb, key=pb.get)
        shared = set(pa) & set(pb)
        delta = max((abs(pa[t] - pb[t]) for t in shared), default=float("nan"))
        worst = max(worst, delta)
        agree += top_a == top_b
        print(f"  {key:>16}  argmax {'==' if top_a == top_b else '!='} "
              f"{top_a!r:12} {top_b!r:12}  {len(shared)} shared, "
              f"max |dlogprob| = {delta:.4f}")
    print(f"  argmax agrees on {agree}/{len(keys)}, "
          f"worst |dlogprob| = {worst:.4f}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--fillers", default=",".join(map(str, FILLERS)),
                    help="comma-separated filler counts (~19 tokens each)")
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        compare(a, b, *args.compare)
        return 0

    assert args.url, "need --url or --compare"
    out = {}
    for filler in map(int, args.fillers.split(",")):
        prompt = build_prompt(filler, 0.5)
        out[f"filler={filler}"] = top_logprobs(args.url, args.model, prompt,
                                               args.n)
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.dump}")
    else:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
