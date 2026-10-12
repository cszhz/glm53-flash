"""Zero-shot GSM8K accuracy against an OpenAI-compatible chat endpoint.

    python3 gsm8k_eval.py --url http://127.0.0.1:8000 --limit 400 --out res.json

Greedy, `reasoning_effort=low` (GLM-5.3 always opens a <think> block; low keeps
it short). The prediction is the last number in the reply; the reference is
the number after `####`. The same --limit/--seed picks the same problems, so
two servers can be compared problem by problem.
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import urllib.request

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--out", required=True)
a = ap.parse_args()

PROMPT = ("Solve the following math problem. Show your reasoning briefly, then "
          "give the final answer as a plain number on the last line in the form "
          "'Answer: <number>'.\n\n{q}")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm(x: str) -> str:
    x = x.replace(",", "").rstrip(".")
    return str(int(float(x))) if re.fullmatch(r"-?\d+(\.0+)?", x) else x


def ask(item):
    i, ex = item
    body = {"model": "GLM-5.3-Flash", "temperature": 0, "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": PROMPT.format(q=ex["question"])}],
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    text = out["choices"][0]["message"]["content"] or ""
    answer = text.split("</think>")[-1]
    m = re.search(r"Answer:\s*\**\s*(" + NUM.pattern + ")", answer)
    nums = NUM.findall(answer)
    pred = norm(m.group(1)) if m else (norm(nums[-1]) if nums else None)
    ref = norm(ex["answer"].split("####")[-1].strip())
    return {"i": i, "ref": ref, "pred": pred, "ok": pred == ref,
            "completion_tokens": out["usage"]["completion_tokens"], "text": text}


ds = load_dataset("openai/gsm8k", "main", split="test")
idx = list(range(len(ds)))
random.Random(a.seed).shuffle(idx)
items = [(i, ds[i]) for i in idx[: a.limit]]
with cf.ThreadPoolExecutor(a.concurrency) as ex:
    res = sorted(ex.map(ask, items), key=lambda r: r["i"])
acc = sum(r["ok"] for r in res) / len(res)
toks = sum(r["completion_tokens"] for r in res) / len(res)
json.dump({"url": a.url, "limit": a.limit, "seed": a.seed, "accuracy": acc,
           "mean_completion_tokens": toks, "results": res}, open(a.out, "w"), indent=1)
print(f"GSM8K {len(res)} problems: accuracy {acc:.4f}  mean completion tokens {toks:.0f}")
