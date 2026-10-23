#!/usr/bin/env python
"""First generated token of server A vs the reference's top-5 at the same prompt.

    python first_token_check.py --a http://10.128.0.184:8000 --ref http://10.128.0.183:8000

Uses only A's *sampled* token (greedy), never A's reported logprobs.
"""
import argparse, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_ab import prompt_ids  # noqa: E402
from tokenizers import Tokenizer


def post(u, b):
    r = urllib.request.Request(u + "/v1/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--lengths", default="1000,2000,2100,3000,4100,6000")
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
tok = Tokenizer.from_file("/home/zzl_google_com/models/GLM-5.3-Flash/tokenizer.json")
for n in map(int, args.lengths.split(",")):
    res = []
    for s in range(args.seeds):
        ids = prompt_ids(tok, n, 50000 + 97 * n + s)
        a = post(args.a, {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 1, "temperature": 0,
                          "logprobs": 0, "return_tokens_as_token_ids": True})
        t = int(a["choices"][0]["logprobs"]["tokens"][0].split(":")[1])
        r = post(args.ref, {"model": "GLM-5.3-Flash", "prompt": ids, "max_tokens": 1, "temperature": 0,
                            "logprobs": 5, "return_tokens_as_token_ids": True})
        top = {int(k.split(":")[1]): v for k, v in r["choices"][0]["logprobs"]["top_logprobs"][0].items()}
        best = max(top.values())
        res.append("argmax" if top.get(t) == best else (f"gap {best - top[t]:.2f}" if t in top else "NOT in top5"))
    print(f"prompt {n:6d}: {res}", flush=True)
