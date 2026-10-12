"""Per-step device time from a vllm-torchtpu trace (one worker's TPU:0).

    python3 prof_summary.py prof/rank_0/*.pt.trace.json.gz --steps 30 [--shapes reshape,copy]

Prints XLA module time per step, then op *self* time (container ops charged
only for what their children do not cover), grouped by op name. `--shapes`
additionally splits the listed op kinds by their HLO signature, which is how
a layout-changing reshape of a big buffer shows up.
"""
import argparse
import collections
import glob
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from trace_load import load_events  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--steps", type=float, required=True)
ap.add_argument("--device", default="/device:TPU:0")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--shapes", default="", help="comma-separated op kinds")
a = ap.parse_args()

paths = [p for pat in a.paths for p in glob.glob(pat)]
ev = load_events(paths[0])
threads = {(e["pid"], e.get("tid")): e["args"].get("name")
           for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
procs = {e["pid"]: e["args"].get("name")
         for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}


def on(e, track):
    return (e.get("ph") == "X" and procs.get(e["pid"]) == a.device
            and threads.get((e["pid"], e.get("tid"))) == track)


mods, mod_n = collections.Counter(), collections.Counter()
for e in ev:
    if on(e, "XLA Modules"):
        n = re.sub(r"\(\d+\)$", "", e["name"])
        mods[n] += e["dur"]
        mod_n[n] += 1
print("== XLA modules (ms/step)")
for n, d in mods.most_common(6):
    print(f"{d / 1e3 / a.steps:8.2f}  x{mod_n[n] / a.steps:.0f}  {n[:100]}")

ops = sorted((e for e in ev if on(e, "XLA Ops")), key=lambda e: (e["ts"], -e["dur"]))
child = [0] * len(ops)
stack = []
for i, e in enumerate(ops):
    while stack and ops[stack[-1]]["ts"] + ops[stack[-1]]["dur"] <= e["ts"]:
        stack.pop()
    if stack:
        child[stack[-1]] += e["dur"]
    stack.append(i)


def kind(name):
    return re.sub(r"[.\d]+$", "", name.split(" = ")[0].lstrip("%"))


self_us, cnt = collections.Counter(), collections.Counter()
shape_us, shape_n = collections.Counter(), collections.Counter()
want = set(filter(None, a.shapes.split(",")))
for i, e in enumerate(ops):
    k = kind(e["name"])
    s = e["dur"] - child[i]
    self_us[k] += s
    cnt[k] += 1
    if k in want:
        sig = re.sub(r"%[\w.\-]+", "%x", e["name"].split(" = ", 1)[-1])[:200]
        shape_us[sig] += s
        shape_n[sig] += 1
total = sum(self_us.values())
print(f"== op self time: {total / 1e3 / a.steps:.2f} ms/step, "
      f"{len(ops) / a.steps:.0f} ops/step")
for k, d in self_us.most_common(a.top):
    print(f"{d / 1e3 / a.steps:8.3f} ms {100 * d / total:5.1f}%  "
          f"x{cnt[k] / a.steps:6.0f}  {k[:90]}")
if want:
    print(f"== by signature: {sorted(want)}")
    for sig, d in shape_us.most_common(12):
        print(f"{d / 1e3 / a.steps:8.3f} ms x{shape_n[sig] / a.steps:5.0f}  {sig}")
