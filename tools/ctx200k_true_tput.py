"""True aggregate throughput per concurrency level from a ctx200k_bench.py JSON.

ctx200k_bench's tok/s/chip is p50(per-request e2e tok/s) * conc / chips, which
overstates throughput once requests queue (the ones that ran first have short
e2e). Here: total (prompt + completion) tokens / wall time of the level.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
chips = d["chips"]
print(f"{'conc':>4} {'wall s':>8} {'tok/s':>9} {'tok/s/chip':>11}")
for case in d["results"]:
    toks = case["ok"] * (case["mean_prompt_tokens"] + case["mean_completion_tokens"])
    wall = case["wall_s"]
    print(f"{case['conc']:4d} {wall:8.1f} {toks / wall:9.0f} {toks / wall / chips:11.1f}")
