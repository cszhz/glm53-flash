#!/usr/bin/env python
"""Is every token the MTP server generated (greedy) the reference model's own
greedy choice, up to numerical near-ties?

    python spec_tf_check.py --spec http://10.128.0.184:8000 --ref http://10.128.0.183:8000

Greedy outputs of two numerically different servers diverge at the first
near-tie and are incomparable afterwards, so instead: generate with MTP, then
teacher-force prompt + that output through the reference server with
the reference's top-5 at every position and check each generated token's rank there. A token that is
not the reference argmax by more than --tie nats is a real error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ab import prompt_ids  # noqa: E402


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--tokenizer",
                    default="/home/zzl_google_com/models/GLM-5.3-Flash/tokenizer.json")
    ap.add_argument("--lengths", default="100,1000,3000,8000,14000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tie", type=float, default=0.15)
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    ok = True
    for i, n in enumerate(int(x) for x in args.lengths.split(",")):
        ids = prompt_ids(tok, n, args.seed * 100 + i)
        out = post(f"{args.spec}/v1/completions", {
            "model": args.model, "prompt": ids, "max_tokens": args.max_tokens,
            "temperature": 0.0, "logprobs": 1, "return_tokens_as_token_ids": True})
        gen = [int(t.split(":", 1)[1]) for t in out["choices"][0]["logprobs"]["tokens"]]
        # prompt_logprobs is not returned by this server, so probe each
        # position: the reference's top-5 next tokens after prompt + gen[:j].
        worst, bad, rank1, probed, big, missing = 0.0, 0, 0, 0, 0, 0
        for j in range(min(len(gen), args.positions)):
            ref = post(f"{args.ref}/v1/completions", {
                "model": args.model, "prompt": ids + gen[:j], "max_tokens": 1,
                "temperature": 0.0, "logprobs": 5,
                "return_tokens_as_token_ids": True})
            top = ref["choices"][0]["logprobs"]["top_logprobs"][0]
            top = {int(k.split(":", 1)[1]): v for k, v in top.items()}
            best = max(top.values())
            mine = top.get(gen[j])
            gap = best - mine if mine is not None else float("inf")
            probed += 1
            rank1 += gap == 0.0
            worst = max(worst, gap)
            bad += gap > args.tie
            big += gap > 0.5
            missing += mine is None
        ok &= bad == 0
        print(f"len {len(ids):6d}: probed {probed} of {len(gen)} tokens, ref argmax "
              f"{rank1}, gap>{args.tie} {bad}, gap>0.5 {big}, not in ref top5 {missing}",
              flush=True)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
