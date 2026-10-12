"""Capture a few prefill steps deep into a long prompt from a running server.

    python3 prof_prefill.py --url http://10.128.0.38:8000 --prefix p100k \
        --filler 4500 --delay 4.0

Sends one long prompt, waits `--delay` s (so the profiled chunks sit near the
end of the prefill, i.e. at long context), starts the profiler (the server's
`active_iterations` decides how many steps it keeps), stops it on the first
token.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from needle_test import build_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--prefix", required=True)
ap.add_argument("--filler", type=int, default=4500)
ap.add_argument("--delay", type=float, default=4.0)
a = ap.parse_args()


def post(path, body=None):
    req = urllib.request.Request(a.url + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def start():
    time.sleep(a.delay)
    post("/start_profile", {"profile_prefix": a.prefix}).read()
    print(f"profiler on at {a.delay:.1f} s")


threading.Thread(target=start, daemon=True).start()
t0 = time.time()
body = {"model": "GLM-5.3-Flash", "prompt": build_prompt(a.filler, 0.5),
        "max_tokens": 2, "temperature": 0, "stream": True}
with post("/v1/completions", body) as r:
    for line in r:
        if line.startswith(b"data: "):
            print(f"first token at {time.time() - t0:.2f} s")
            break
time.sleep(max(0.0, a.delay + 0.5 - (time.time() - t0)))
post("/stop_profile").read()
