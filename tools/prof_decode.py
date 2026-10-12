"""Capture a decode-only profile window from a running vLLM server.

Starts a streaming request, turns the profiler on once tokens are flowing
(so the prefill is not in the trace), and turns it off after N more tokens.

    python3 prof_decode.py --prefix decode_sparse --filler 120 --steps 30
"""
import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from needle_test import build_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--prefix", required=True)
ap.add_argument("--filler", type=int, default=120)
ap.add_argument("--skip", type=int, default=8)
ap.add_argument("--steps", type=int, default=30)
a = ap.parse_args()


def post(path, body=None):
    req = urllib.request.Request(a.url + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


body = {"model": "GLM-5.3-Flash", "prompt": build_prompt(a.filler, 0.5),
        "max_tokens": a.skip + a.steps + 16, "temperature": 0,
        "ignore_eos": True, "stream": True}
n, t_on, t_off = 0, None, None
with post("/v1/completions", body) as r:
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        n += 1
        if n == a.skip:
            post("/start_profile", {"profile_prefix": a.prefix}).read()
            t_on = time.time()
        elif n == a.skip + a.steps:
            t_off = time.time()
            post("/stop_profile").read()
print(f"tokens={n} profiled_steps={a.steps} "
      f"wall_per_step={(t_off - t_on) / a.steps * 1e3:.1f} ms (includes profiler overhead)")
