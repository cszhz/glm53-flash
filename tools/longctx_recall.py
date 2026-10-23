#!/usr/bin/env python
"""Long-context recall: exact-answer lookups in prompts far past index_topk.

    python longctx_recall.py --url ref=http://10.128.0.183:8000 --url mtp=http://10.128.0.184:8000

Each prompt lists N maintenance records ("node i closes at HH:MM") and asks
for one node's time; the answer is computable, so each server gets a plain
accuracy. Prompts run ~3.3K-13.4K tokens, so the DSA indexer has to pick the
right pools (GSM8K prompts are too short to exercise it).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import urllib.request


def record_time(i: int) -> str:
    return f"{(i * 7 + i // 5) % 24:02d}:{(i * 13 + i // 3) % 60:02d}"


def build(n: int, target: int) -> str:
    lines = [f"Record {i:04d}: the maintenance window for cluster node {i} "
             f"closes at {record_time(i)} UTC." for i in range(n)]
    return ("\n".join(lines) + f"\n\nAt what time does the maintenance window "
            f"for cluster node {target} close? Reply with the time only, as HH:MM.")


def ask(url: str, model: str, prompt: str) -> str:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 256,
                       "reasoning_effort": "low",
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0]["message"].get("content") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True, help="name=url")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    cases = []
    for _ in range(args.n):
        n = rnd.randint(150, 600)
        cases.append((n, rnd.randrange(n)))
    for name, url in (u.split("=", 1) for u in args.url):
        def one(case):
            n, t = case
            m = re.findall(r"\d{1,2}:\d{2}", ask(url, args.model, build(n, t)))
            return bool(m) and m[-1].zfill(5) == record_time(t)
        with cf.ThreadPoolExecutor(args.concurrency) as ex:
            ok = list(ex.map(one, cases))
        print(f"{name}: {sum(ok)}/{len(ok)} correct; wrong cases "
              f"{[cases[i] for i, x in enumerate(ok) if not x][:10]}", flush=True)


if __name__ == "__main__":
    main()
