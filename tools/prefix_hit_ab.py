#!/usr/bin/env python
"""Prefix-cache hit vs miss on the same prompt: outputs must not change.

    python prefix_hit_ab.py --url http://10.128.0.184:8000 [--block 2048]
    python prefix_hit_ab.py --url ... --mode share       # needs VLLM_SERVER_DEV_MODE=1
    python prefix_hit_ab.py --url ... --mode concurrent  # needs VLLM_SERVER_DEV_MODE=1

`share`: P+S2 served on top of P cached by a different request (P+S1) must
match P+S2 computed from an empty cache. `concurrent`: requests sharing
prefixes, all in flight at once, must each match their own run from an empty
cache.

For each prompt length (around the cache block boundary, and long enough
that DSA selection is active), send a prompt with a fresh leading nonce
(a miss), then the identical token ids again (a hit on every whole block
before the last token), and compare the greedy tokens and their logprobs.
The server's prefix-cache counters (/metrics) confirm the second request
actually hit.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request

from tokenizers import Tokenizer

WORDS = ("ledger pallet aisle scan truck inspector warehouse inbound outbound "
         "crate invoice forklift dock shelf barcode shipment").split()


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def hits(url: str) -> tuple[float, float]:
    text = urllib.request.urlopen(f"{url}/metrics", timeout=60).read().decode()

    def total(name):
        return sum(float(m) for m in re.findall(
            rf"^vllm:{name}_total\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M))

    return total("prefix_cache_hits"), total("prefix_cache_queries")


def prompt_ids(tok: Tokenizer, n: int, seed: int) -> list[int]:
    rnd = random.Random(seed)
    head = f"Record {seed}: summarize the ledger below and quote line {seed % 97}.\n"
    ids = tok.encode(head).ids
    line = 0
    while len(ids) < n:
        words = " ".join(rnd.choice(WORDS) for _ in range(12))
        ids += tok.encode(f"line {line}: {words}\n").ids
        line += 1
    return ids[:n]


def reset(url: str) -> None:
    req = urllib.request.Request(f"{url}/reset_prefix_cache", data=b"",
                                 method="POST")
    urllib.request.urlopen(req, timeout=600).read()


def compare(label, ref, got, need_hits=None, hit=None) -> bool:
    (rt, rl), (gt, gl) = ref, got
    same = rt == gt
    # Logprobs are only comparable up to the first differing token.
    n = next((j for j, (a, c) in enumerate(zip(rt, gt)) if a != c), len(rt))
    diff = max((abs(a - c) for a, c in zip(rl[:n + 1], gl[:n + 1])),
               default=0.0)
    good = same and diff < 0.05 and (need_hits is None or hit >= need_hits)
    hit_s = "" if need_hits is None else f"hits {hit:7.0f} (expect >= {need_hits}) | "
    print(f"{label}: {hit_s}tokens equal {same}"
          f"{'' if same else f' (first diff at {n})'} | max |dlogprob| "
          f"through first diff {diff:.4f} | "
          f"{'OK' if good else 'MISMATCH'}", flush=True)
    return good


def run_share(args, tok) -> bool:
    b, ok = args.block, True
    for i, k in enumerate([b + 5, 3 * b + 100, 7 * b + 1, 20 * b + 77,
                           60 * b + 500]):
        p = prompt_ids(tok, k, seed=args.seed * 7000 + i)
        s1 = prompt_ids(tok, 400, seed=args.seed * 7000 + 500 + i)[5:]
        s2 = prompt_ids(tok, 400, seed=args.seed * 7000 + 900 + i)[5:]
        reset(args.url)
        fresh = generate(args.url, args.model, p + s2, args.max_tokens)
        reset(args.url)
        generate(args.url, args.model, p + s1, args.max_tokens)
        h0, _ = hits(args.url)
        shared = generate(args.url, args.model, p + s2, args.max_tokens)
        h1, _ = hits(args.url)
        ok &= compare(f"share prefix {k:7d}", fresh, shared, k // b * b,
                      h1 - h0)
    return ok


def run_concurrent(args, tok) -> bool:
    from concurrent.futures import ThreadPoolExecutor
    b = args.block
    bases = [prompt_ids(tok, n, seed=args.seed * 9000 + j)
             for j, n in enumerate([3 * b + 11, 9 * b + 300])]
    prompts = []
    for j in range(16):
        base = bases[j % 2][: (j % 4 + 1) * b + 17 * j]
        if args.unique_prefix:
            # Control: same lengths and batching, but no shared prefix.
            base = prompt_ids(tok, len(base), seed=args.seed * 9300 + j)
        prompts.append(base + prompt_ids(tok, 200 + 37 * j,
                                         seed=args.seed * 9100 + j)[5:])
    refs = []
    for ids in prompts:
        reset(args.url)
        refs.append(generate(args.url, args.model, ids, args.max_tokens))
    reset(args.url)
    ok = True
    for rnd in range(2):
        with ThreadPoolExecutor(len(prompts)) as pool:
            outs = list(pool.map(lambda ids: generate(
                args.url, args.model, ids, args.max_tokens), prompts))
        for j, (ref, out) in enumerate(zip(refs, outs)):
            ok &= compare(f"concurrent round {rnd} req {j:2d} "
                          f"len {len(prompts[j]):6d}", ref, out)
    return ok


def generate(url: str, model: str, ids: list[int], max_tokens: int):
    out = post(f"{url}/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": max_tokens,
        "temperature": 0.0, "logprobs": 1, "ignore_eos": True,
    })
    lp = out["choices"][0]["logprobs"]
    return lp["tokens"], lp["token_logprobs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--tokenizer",
                    default="/gcs/models/GLM-5.3-Flash/tokenizer.json")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--lengths", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unique-prefix", action="store_true",
                    help="concurrent mode control: no shared prefixes")
    ap.add_argument("--mode", default="same",
                    choices=["same", "share", "concurrent"])
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    if args.mode != "same":
        fn = run_share if args.mode == "share" else run_concurrent
        print("PASS" if fn(args, tok) else "FAIL")
        return
    b = args.block
    lengths = ([int(x) for x in args.lengths.split(",")] if args.lengths else
               [b - 3, b - 1, b, b + 1, b + 3, 2 * b + 5, 3 * b + 1000,
                10 * b + 7, 40 * b + 123])
    ok = True
    for i, n in enumerate(lengths):
        ids = prompt_ids(tok, n, seed=args.seed * 1000 + i)
        h0, q0 = hits(args.url)
        miss_tokens, miss_lp = generate(args.url, args.model, ids,
                                        args.max_tokens)
        h1, q1 = hits(args.url)
        hit_tokens, hit_lp = generate(args.url, args.model, ids,
                                      args.max_tokens)
        h2, q2 = hits(args.url)
        expect_hit = (n - 1) // b * b
        same = miss_tokens == hit_tokens
        diff = max(abs(a - c) for a, c in zip(miss_lp, hit_lp))
        first_div = next((j for j, (a, c) in enumerate(
            zip(miss_tokens, hit_tokens)) if a != c), None)
        good = same and diff < 0.05 and (h2 - h1) >= expect_hit
        ok &= good
        print(f"len {n:7d}: miss-run hits {h1 - h0:7.0f}, hit-run hits "
              f"{h2 - h1:7.0f} (expect >= {expect_hit}) | tokens equal {same}"
              f"{'' if same else f' (first diff at {first_div})'} | max "
              f"|dlogprob| {diff:.4f} | {'OK' if good else 'MISMATCH'}",
              flush=True)
        if not same:
            print("   miss:", "".join(miss_tokens)[:200].replace("\n", " "))
            print("   hit :", "".join(hit_tokens)[:200].replace("\n", " "))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
