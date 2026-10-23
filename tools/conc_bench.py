"""Concurrency ladder: N parallel streaming requests, TTFT / TPOT / throughput.

    python3 conc_bench.py --url http://127.0.0.1:8000 --filler 340 --out 256 --conc 1,2,4

Each level sends `--rounds * conc` requests, `conc` at a time, each with a
distinct prompt (needle at a different depth) so nothing can be shared.
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from needle_test import build_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--filler", type=int, default=340)
ap.add_argument("--out", type=int, default=256)
ap.add_argument("--conc", default="1,2,4")
ap.add_argument("--rounds", type=int, default=2)
a = ap.parse_args()


def one(i: int):
    body = {"model": "GLM-5.3-Flash", "prompt": build_prompt(a.filler, (i % 97) / 97),
            "max_tokens": a.out, "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            if not ev.get("choices"):
                continue
            n += 1
            if first is None:
                first = time.time()
    end = time.time()
    # Under speculative decoding one event can carry several tokens, so count
    # the server's completion tokens, not events.
    if usage and usage.get("completion_tokens"):
        n = usage["completion_tokens"]
    return first - t0, (end - first) / max(n - 1, 1), n


print(f"filler={a.filler} out={a.out}")
print(f"{'conc':>4} {'TTFT p50 s':>11} {'TPOT p50 ms':>12} {'out tok/s':>10} {'tok/s/chip':>11}")
seq = 0
for conc in map(int, a.conc.split(",")):
    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        futs = [ex.submit(one, seq + i) for i in range(conc * a.rounds)]
        results = [f.result() for f in futs]
    wall = time.time() - t0
    seq += conc * a.rounds
    toks = sum(r[2] for r in results)
    print(f"{conc:4d} {statistics.median(r[0] for r in results):11.3f} "
          f"{statistics.median(r[1] for r in results) * 1e3:12.2f} "
          f"{toks / wall:10.1f} {toks / wall / 4:11.1f}")
