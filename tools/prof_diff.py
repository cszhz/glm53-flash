"""Per-call op time of one XLA module in two traces, sorted by estimated cost.

    python3 prof_diff.py <trace_a.gz> <trace_b.gz> [--index-a 0] [--index-b 0]

The windows need not cover the same number of layers: each op's time per call
is compared, and the delta is scaled by trace B's call count.
"""
import argparse
import collections
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from trace_load import load_events  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--index-a", type=int, default=0)
ap.add_argument("--index-b", type=int, default=0)
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()


def per_op(path, index):
    ev = load_events(path)
    th = {(e["pid"], e.get("tid")): e["args"].get("name") for e in ev
          if e.get("ph") == "M" and e.get("name") == "thread_name"}
    pn = {e["pid"]: e["args"].get("name") for e in ev
          if e.get("ph") == "M" and e.get("name") == "process_name"}
    dev = "/device:TPU:0"
    mods = sorted((e for e in ev if e.get("ph") == "X" and pn.get(e["pid"]) == dev
                   and th.get((e["pid"], e.get("tid"))) == "XLA Modules"
                   and e["dur"] > 1e5), key=lambda e: e["ts"])
    m = mods[index]
    lo, hi = m["ts"], m["ts"] + m["dur"]
    ops = sorted((e for e in ev if e.get("ph") == "X" and pn.get(e["pid"]) == dev
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
    t, n = collections.Counter(), collections.Counter()
    for i, e in enumerate(ops):
        k = re.sub(r"\.\d+", "", e["name"].split(" = ")[0].lstrip("%"))[:60]
        t[k] += e["dur"] - child[i]
        n[k] += 1
    return m["dur"], t, n


da, ta, na = per_op(args.a, args.index_a)
db, tb, nb = per_op(args.b, args.index_b)
print(f"module A {da / 1e3:.1f} ms, B {db / 1e3:.1f} ms")
rows = []
for k in set(ta) | set(tb):
    pa = ta[k] / na[k] if na[k] else None
    pb = tb[k] / nb[k] if nb[k] else None
    if pa is None or pb is None:
        rows.append((tb[k] - 0, k, pa, pb, nb[k]))
        continue
    rows.append(((pb - pa) * nb[k], k, pa, pb, nb[k]))
rows.sort(key=lambda r: -abs(r[0]))
print(f"{'delta@B(ms)':>11} {'A/call(us)':>11} {'B/call(us)':>11} {'ratio':>6} {'calls B':>7}  op")
for d, k, pa, pb, c in rows[:args.top]:
    ratio = f"{pb / pa:6.2f}" if pa and pb else "   new" if pa is None else "  gone"
    print(f"{d / 1e3:11.2f} {(pa or 0):11.1f} {(pb or 0):11.1f} {ratio} {c:7d}  {k}")
