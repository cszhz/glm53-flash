"""Op self time inside one XLA module execution, by source jit and by op.

    python3 prof_window.py <trace.json.gz> --index 1 [--top 30]

`--index` picks the module among those longer than 100 ms, in time order.
"""
import argparse
import collections
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from trace_load import load_events  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--device", default="/device:TPU:0")
a = ap.parse_args()
ev = load_events(a.path)
th = {(e["pid"], e.get("tid")): e["args"].get("name") for e in ev
      if e.get("ph") == "M" and e.get("name") == "thread_name"}
pn = {e["pid"]: e["args"].get("name") for e in ev
      if e.get("ph") == "M" and e.get("name") == "process_name"}
mods = sorted((e for e in ev if e.get("ph") == "X" and pn.get(e["pid"]) == a.device
               and th.get((e["pid"], e.get("tid"))) == "XLA Modules"
               and e["dur"] > 1e5), key=lambda e: e["ts"])
m = mods[a.index]
lo, hi = m["ts"], m["ts"] + m["dur"]
print(f"module {a.index}: {m['dur'] / 1e3:.1f} ms")
ops = sorted((e for e in ev if e.get("ph") == "X" and pn.get(e["pid"]) == a.device
              and th.get((e["pid"], e.get("tid"))) == "XLA Ops"
              and lo <= e["ts"] < hi), key=lambda e: (e["ts"], -e["dur"]))
child = [0] * len(ops)
st = []
for i, e in enumerate(ops):
    while st and ops[st[-1]]["ts"] + ops[st[-1]]["dur"] <= e["ts"]:
        st.pop()
    if st:
        child[st[-1]] += e["dur"]
    st.append(i)
by_src, by_op, n = collections.Counter(), collections.Counter(), collections.Counter()
for i, e in enumerate(ops):
    s = re.search(r'tf_op": "([^"]*)', str(e.get("args", {})))
    tf = s.group(1) if s else ""
    j = re.findall(r"jit\(([^)]*)\)", tf)
    key = j[0] if j else (tf.split("/")[0][:40] or "?")
    self_t = e["dur"] - child[i]
    by_src[key] += self_t
    op = re.sub(r"[.\d]+$", "", e["name"].split(" = ")[0].lstrip("%"))[:60]
    by_op[op] += self_t
    n[op] += 1
tot = sum(by_src.values())
print(f"self total {tot / 1e3:.1f} ms, idle {(m['dur'] - tot) / 1e3:.1f} ms")
print("-- by source")
for k, v in by_src.most_common(a.top):
    print(f"{v / 1e3:8.2f} ms {100 * v / tot:5.1f}%  {k[:80]}")
print("-- by op")
for k, v in by_op.most_common(a.top):
    print(f"{v / 1e3:8.2f} ms {100 * v / tot:5.1f}% x{n[k]:5d}  {k}")
