#!/usr/bin/env python3
"""GLM-5.3-Flash 200K 长上下文压测（TPU 侧），口径对齐 h200/result.txt 的 GPU 基线。

平时用 bench_ctx200k.sh 调，不直接敲。

== 为什么要单独一个文件，不复用 bench.sh ==
bench.sh 是 1k 进 / 1k 出的冒烟档，走 sgl_jax.bench_serving + random-ids，
只回答「跑不跑得动」。GPU 基线是 200K 进 / 128 出、OpenAI 流式、并发梯度，
两者的输入长度、语料、接口、并发形态没有一项相同，数放在一张表里是错的。

== 和 GPU 那份 ctx200k_bench.py 的关系 ==
**指标定义逐字照抄**，这样两边的数才能直接比。照抄的部分：
  - FILLER 语料、build_user_text 的构造和二分截断，一字不差
  - 请求体：/v1/chat/completions 流式、temperature=0、ignore_eos、
    stream_options.include_usage、chat_template_kwargs 关思考
  - TTFT = 第一个**非空 content 分片**的时刻（不是第一个 SSE 分片）
  - gen = e2e - ttft；decode_tps = completion/gen；tpot = gen/completion
    （注意分母是 completion，不是 completion-1 —— GPU 那份就是这么算的，
      跟着错才对得上）
  - extend_tps = prompt/ttft；e2e_tps = (prompt+completion)/e2e
  - pct() 的线性插值百分位，连实现都一样
  - case 之间 flush_cache，每条 prompt 带唯一 nonce（冷前缀）

TPU 这边不一样的地方，只有这些：
  - tokenizer 直接 Tokenizer.from_file(tokenizer.json)。GLM-5.3 的
    tokenizer_config.json 写着 transformers 5.0 才有的 TokenizersBackend，
    走 AutoTokenizer 会炸（bench.sh 为此维护了一个补丁目录）；裸 tokenizers
    库读 tokenizer.json 没这个问题，而且和 GPU 那边的加载方式恰好一致。
  - 芯片数从服务端 /get_server_info 的 tp_size 反推（v7x 每 chip 2 core，
    所以 chips = tp_size/2），不写死。tok/s/chip 这一列靠它。
  - 并发梯度加了 16（GPU 那份跑到 12）。
  - 并发 1 那档发 3 条，对齐 GPU 表里「Requests 3 / Concurrency 1」那行；
    其余档 n == conc，和 GPU 一样一波打完。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tokenizers import Tokenizer

# 和 GPU 那份逐字相同。改这里 = 两边不可比。
FILLER = (
    "The warehouse inventory ledger records pallet identifiers, aisle coordinates, "
    "timestamped inbound scans, outbound truck numbers, and inspector notes. "
)

# 并发梯度。GPU 跑的是 1/2/4/8/12，这里按要求加一档 16。
LADDER = [1, 2, 4, 8, 12, 16]
# 每档发多少条。并发 1 发 3 条是为了对齐 GPU 表首行（Requests 3 / Concurrency 1）；
# 只发 1 条的话 p50/p90 都退化成同一个数，没有分布可言。
REQUESTS_AT = {1: 3}


def http_json(url: str, data: bytes | None = None, timeout: float = 30.0) -> tuple[int, object]:
    req = Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else {}


def healthy(base: str, timeout: float = 10.0) -> bool:
    try:
        status, body = http_json(base + "/v1/models", timeout=timeout)
        return status == 200 and isinstance(body, dict) and bool(body.get("data"))
    except Exception:
        return False


def flush_cache(base: str, timeout: float = 60.0) -> None:
    # 这一档服务端是 --disable-radix-cache 起的，本来就没有前缀缓存可刷；
    # 照发一遍纯粹是和 GPU 的流程对齐，失败不算错。
    for path in ("/flush_cache", "/v1/flush_cache"):
        try:
            http_json(base + path, data=b"{}", timeout=timeout)
            return
        except Exception:
            continue


def server_info(base: str) -> dict:
    """读服务端真实 argv。不信客户端 shell 里 export 的值 —— 那些是服务端
    **启动时**读的，起好之后在这边改无效且不报错。"""
    try:
        _, body = http_json(base + "/get_server_info", timeout=15.0)
    except Exception as exc:
        print(f"  ! 读不到 /get_server_info: {exc}", flush=True)
        return {}
    if not isinstance(body, dict):
        return {}
    return body.get("server_args", body)


def served_model(base: str, fallback: str) -> str:
    try:
        _, body = http_json(base + "/v1/models", timeout=15.0)
        rows = (body or {}).get("data") or []
        if rows and rows[0].get("id"):
            return rows[0]["id"]
    except Exception:
        pass
    return fallback


def build_user_text(tokenizer: Tokenizer, target_tokens: int, nonce: str) -> str:
    """和 GPU 那份逐字相同（含二分截断）。别顺手"优化"，改了就不可比。"""
    prefix = f"BENCH nonce={nonce} target={target_tokens}. Read the ledger and end with the nonce.\n"
    prefix_n = len(tokenizer.encode(prefix).ids)
    block_ids = tokenizer.encode(FILLER).ids
    if not block_ids:
        raise RuntimeError("empty filler tokenization")
    remain = max(1, target_tokens - prefix_n)
    reps = max(1, (remain + len(block_ids) - 1) // len(block_ids))
    text = prefix + FILLER * reps
    n = len(tokenizer.encode(text).ids)
    if n > target_tokens:
        extra = n - target_tokens
        drop = extra // len(block_ids)
        if drop:
            text = prefix + FILLER * max(1, reps - drop)
            n = len(tokenizer.encode(text).ids)
    if n < target_tokens:
        text += FILLER * (((target_tokens - n) + len(block_ids) - 1) // len(block_ids))
        n = len(tokenizer.encode(text).ids)
    if n > target_tokens:
        lo, hi = len(prefix), len(text)
        best = text
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[:mid]
            cn = len(tokenizer.encode(cand).ids)
            if cn >= target_tokens:
                best = cand
                hi = mid - 1
            else:
                lo = mid + 1
        text = best
    return text


def one_request(base: str, model: str, user_text: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    payload = json.dumps(body).encode()
    req = Request(
        base + "/v1/chat/completions",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    content: list[str] = []
    usage: dict = {}
    finish = None
    try:
        with urlopen(req, timeout=timeout) as resp:
            buf = b""
            while True:
                piece = resp.read(4096)
                if not piece:
                    break
                buf += piece
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("usage"):
                        usage = ev["usage"]
                    choices = ev.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token_text = delta.get("content") or delta.get("reasoning_content") or ""
                    if token_text:
                        # TTFT 认第一个**非空**分片：服务端会先发一个只带 role 的
                        # 空 delta，拿它当首 token 会把 TTFT 报小。
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                        content.append(token_text)
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
        e2e = time.perf_counter() - t0
        prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
        completion_tokens = int((usage or {}).get("completion_tokens") or 0)
        # usage 缺失时的兜底：completion 用收到的分片数（stream-interval=1，
        # 一个分片一个 token），prompt 用 0 并在汇总里标出来，不硬凑一个假数。
        usage_ok = bool(usage) and prompt_tokens > 0
        if not completion_tokens:
            completion_tokens = chunks
        details = (usage or {}).get("prompt_tokens_details") or {}
        cached_tokens = 0
        if isinstance(details, dict):
            cached_tokens = int(details.get("cached_tokens") or 0)
        if isinstance(usage, dict) and usage.get("cached_tokens") is not None:
            cached_tokens = int(usage.get("cached_tokens") or 0)
        if ttft is None:
            ttft = e2e
        gen = max(e2e - ttft, 1e-6)
        return {
            "ok": True,
            "error": None,
            "usage_ok": usage_ok,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "ttft_s": ttft,
            "e2e_s": e2e,
            # 下面四个就是 GPU 表那几列的原料，公式逐字照抄。
            "extend_tps": prompt_tokens / ttft if ttft > 0 else 0.0,
            "decode_tps": (completion_tokens / gen) if completion_tokens else 0.0,
            "tpot_s": (gen / completion_tokens) if completion_tokens else None,
            "e2e_tps": ((prompt_tokens + completion_tokens) / e2e) if e2e > 0 else 0.0,
            "chunks": chunks,
            "finish_reason": finish,
            "text_head": "".join(content)[:120],
        }
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        err = str(exc)
        if isinstance(exc, HTTPError):
            try:
                err = f"{exc} {exc.read()[:400]!r}"
            except Exception:
                pass
        return {
            "ok": False,
            "error": err,
            "usage_ok": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "ttft_s": None,
            "e2e_s": time.perf_counter() - t0,
            "extend_tps": 0.0,
            "decode_tps": 0.0,
            "tpot_s": None,
            "e2e_tps": 0.0,
            "chunks": 0,
            "finish_reason": None,
            "text_head": "",
        }


def pct(vals: list[float], p: float) -> float | None:
    """线性插值百分位。和 GPU 那份同一份实现 —— 百分位算法不同能差出几个点，
    对比表里这点差异会被当成性能差异读。"""
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def run_case(args, tokenizer, model: str, target_in: int, out_len: int, conc: int, n: int,
             name: str) -> dict:
    print(f"=== {name}  in≈{target_in} out={out_len} conc={conc} n={n} ===", flush=True)
    flush_cache(args.base)
    print("  构造 prompt（200K 要编码一会）...", flush=True)
    jobs = []
    for i in range(n):
        nonce = f"{name}-{i}-{uuid.uuid4().hex[:8]}"
        jobs.append(build_user_text(tokenizer, target_in, nonce))
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one_request, args.base, model, text, out_len, args.timeout)
                for text in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            status = "ok" if row["ok"] else "FAIL"
            ttft = f"{row['ttft_s']:.2f}" if row["ttft_s"] is not None else "-"
            print(f"  {status} prompt={row['prompt_tokens']} out={row['completion_tokens']} "
                  f"ttft={ttft}s e2e={row['e2e_s']:.2f}s "
                  f"extend={row['extend_tps']:.0f} decode={row['decode_tps']:.1f}"
                  + (f" err={row['error']}" if not row["ok"] else ""), flush=True)
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r["ok"]]

    def p(key, q):
        return pct([r[key] for r in ok if r.get(key) is not None], q)

    e2e_tps_p50 = p("e2e_tps", 0.50)
    summary = {
        "name": name,
        "n": len(rows),
        "conc": conc,
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "wall_s": wall,
        "usage_missing": sum(1 for r in ok if not r["usage_ok"]),
        "mean_prompt_tokens": statistics.mean([r["prompt_tokens"] for r in ok]) if ok else None,
        "mean_completion_tokens": (
            statistics.mean([r["completion_tokens"] for r in ok]) if ok else None),
        "mean_cached_tokens": statistics.mean([r["cached_tokens"] for r in ok]) if ok else None,
        "ttft_p50_s": p("ttft_s", 0.50), "ttft_p90_s": p("ttft_s", 0.90),
        "tpot_p50_s": p("tpot_s", 0.50), "tpot_p90_s": p("tpot_s", 0.90),
        "extend_tps_p50": p("extend_tps", 0.50),
        "decode_tps_p50": p("decode_tps", 0.50),
        "e2e_tps_p50": e2e_tps_p50, "e2e_tps_p90": p("e2e_tps", 0.90),
        "e2e_lat_p50_s": p("e2e_s", 0.50), "e2e_lat_p90_s": p("e2e_s", 0.90),
        # GPU 表那一列：单请求 E2E 吞吐 × 并发 / 芯片数。我按 h200/result.txt
        # 每一行反推验过这个公式（并发 1 那行差 0.8%，像是用了 mean 不是 p50）。
        # 注意它是 **prefill 主导**的数：199700 进对 128 出，输出只占 0.06%。
        "tok_s_chip": (e2e_tps_p50 * conc / args.chips) if e2e_tps_p50 and args.chips else None,
        "rows": rows,
    }
    print(f"  小结 ok={summary['ok']}/{summary['n']} "
          f"ttft_p50={_ms(summary['ttft_p50_s'])}ms tpot_p50={_ms(summary['tpot_p50_s'])}ms "
          f"tok/s/chip={_f(summary['tok_s_chip'])} wall={wall:.1f}s "
          f"healthy={healthy(args.base)}", flush=True)
    if summary["usage_missing"]:
        print(f"  ! {summary['usage_missing']} 条没拿到 usage，prompt_tokens 会是 0，"
              f"extend/e2e 吞吐那几列不可信", flush=True)
    return summary


def _ms(v):
    return f"{v * 1000:.0f}" if v is not None else "-"


def _f(v, nd=1):
    return f"{v:.{nd}f}" if v is not None else "-"


# 列名和 h200/result.txt 逐字相同，两份 TSV 能直接上下拼。
HEADER = ["Model", "Framework", "Topo", "Context Length", "Requests", "Concurrency",
          "Input/Output", "TTFT p50 (ms)", "TTFT p90 (ms)", "TPOT p50 (ms/tok)",
          "TPOT p90 (ms/tok)", "Extend tput p50 (tok/s)", "Decode tput p50 (tok/s)",
          "tok/s/chip", "E2E tput p50 (tok/s)", "E2E tput p90 (tok/s)",
          "E2E latency p50 (s)", "E2E latency p90 (s)"]


def tsv_row(s: dict, args) -> list[str]:
    io = (f"{int(s['mean_prompt_tokens'])}/{int(s['mean_completion_tokens'])}"
          if s["mean_prompt_tokens"] is not None else "-")
    return [
        args.model_label, "sglang-jax", args.topo, args.ctx_label,
        str(s["n"]), str(s["conc"]), io,
        _ms(s["ttft_p50_s"]), _ms(s["ttft_p90_s"]),
        _f(s["tpot_p50_s"] * 1000 if s["tpot_p50_s"] else None, 2),
        _f(s["tpot_p90_s"] * 1000 if s["tpot_p90_s"] else None, 2),
        _f(s["extend_tps_p50"], 0), _f(s["decode_tps_p50"], 1), _f(s["tok_s_chip"], 1),
        _f(s["e2e_tps_p50"], 0), _f(s["e2e_tps_p90"], 0),
        _f(s["e2e_lat_p50_s"], 2), _f(s["e2e_lat_p90_s"], 2),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="服务地址，如 http://10.128.0.13:30000")
    ap.add_argument("--model", default="", help="留空则从 /v1/models 读")
    ap.add_argument("--model-label", default="GLM-5.3-Flash-FP8", help="表里 Model 列")
    ap.add_argument("--tokenizer", default="/gcs/models/GLM-5.3-Flash/tokenizer.json")
    ap.add_argument("--out", required=True, help="JSON 明细")
    ap.add_argument("--out-tsv", default="", help="TSV 汇总，默认 --out 同名 .tsv")
    ap.add_argument("--target-in", type=int, default=200000)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--ladder", default=",".join(str(c) for c in LADDER))
    ap.add_argument("--chips", type=int, default=0, help="0 = 从 /get_server_info 的 tp_size 推")
    ap.add_argument("--timeout", type=float, default=3600,
                    help="单请求超时。TPU 侧 200K prefill 比 GPU 慢，"
                         "GPU 那份默认的 1200s 在高并发档会不够")
    ap.add_argument("--warmup", action="store_true", help="先发一条 2k 的热身（不计入结果）")
    args = ap.parse_args()

    if not healthy(args.base):
        print(f"错误：{args.base} 未就绪。先 bash start_slice.sh", file=sys.stderr)
        return 1

    info = server_info(args.base)
    attn = info.get("attention_backend", "?")
    tp = info.get("tp_size") or 0
    ep = info.get("ep_size") or tp
    ctx = info.get("context_length") or 0
    # v7x 每 chip 2 core，JAX device 数 = tp_size，所以芯片数是它的一半。
    if not args.chips:
        args.chips = (tp // 2) if tp else 0
    args.topo = f"{args.chips}×TPUv7x TP{tp}/EP{ep}" if args.chips else f"TP{tp}/EP{ep}"
    args.ctx_label = f"{args.target_in // 1000}K"

    print(f"── 服务 {args.base}")
    print(f"   attn={attn} tp={tp} ep={ep} ctx={ctx} kv={info.get('kv_cache_dtype', '?')}")
    print(f"   芯片数 {args.chips}（tok/s/chip 的分母）  Topo={args.topo}")

    if attn == "dsa_sparse":
        # indexer 已经通了（token 级 top-k），但 k-pooling 还没实现，选出来的
        # 不是训练时那套 pool 级选择。速度数可以看，精度必须另外过门。
        print("注意：dsa_sparse 走的是 token 级 top-k（k-pooling 未实现），"
              "精度需另行验证，别把这个数当成已验证配置。", file=sys.stderr)
    if ctx and args.target_in + args.out_len > ctx:
        print(f"错误：要 {args.target_in}+{args.out_len} tokens，但服务端 context_length={ctx}。",
              file=sys.stderr)
        return 1
    # 不拦，只提醒：这两个值决定 200K 会被切成多少个 chunk 去 prefill，
    # 直接影响 TTFT 和 Extend tput。和 GPU 比之前得知道这一条。
    cps = info.get("chunked_prefill_size") or 0
    mpt = info.get("max_prefill_tokens") or 0
    if cps and args.target_in > cps:
        print(f"   ! chunked_prefill_size={cps} max_prefill_tokens={mpt}："
              f"{args.target_in} 的 prompt 要切约 {args.target_in // cps} 个 chunk 做 prefill。")
        # 0922：这里原来无条件写「这是冒烟档配置」。16384 恰恰**就是** 200K 出厂档
        # （见 exp-prof-0915/logs_tuned/10.128.0.13.log 里服务端自己打的 ServerArgs），
        # 所以那句话在正确配置下反而在报假警，得按值判。
        if cps == 16384 and mpt == 16384:
            print(f"     16384 = 0915 出厂 200K 档，和历史梯度可比。")
        else:
            print(f"     出厂 200K 档是 16384/16384，当前不是这一档 —— TTFT / Extend tput "
                  f"会因此和历史梯度不可比。先确认这是你要的口径。")

    model = args.model or served_model(args.base, "default")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ladder = [int(x) for x in args.ladder.split(",") if x.strip()]

    results = []
    if args.warmup:
        run_case(args, tokenizer, model, 2048, 32, 1, 1, "warmup_2k")

    for conc in ladder:
        n = REQUESTS_AT.get(conc, conc)
        s = run_case(args, tokenizer, model, args.target_in, args.out_len, conc, n,
                     f"ctx{args.ctx_label}_c{conc}")
        results.append(s)
        # 和 GPU 一样的中止条件：服务挂了或这一档挂掉四分之一就不再加压。
        if not healthy(args.base):
            print(f"停在 {s['name']}：服务已不健康", flush=True)
            break
        if s["n"] and s["failed"] / s["n"] >= 0.25:
            print(f"停在 {s['name']}：失败率 {s['failed']}/{s['n']}", flush=True)
            break

    payload = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": args.base, "model": model, "model_label": args.model_label,
        "topo": args.topo, "chips": args.chips,
        "target_in": args.target_in, "out_len": args.out_len,
        "server_args": info,
        "notes": ("200K 长上下文基线，口径对齐 h200/result.txt："
                  "thinking=false, ignore_eos, 唯一 nonce, case 间 flush_cache, "
                  "每档一波 n==conc（并发 1 档发 3 条）。"
                  "指标公式与 GPU 侧 ctx200k_bench.py 逐字相同。"),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    tsv_path = args.out_tsv or str(Path(args.out).with_suffix(".tsv"))
    lines = ["\t".join(HEADER)] + ["\t".join(tsv_row(s, args)) for s in results]
    Path(tsv_path).write_text("\n".join(lines) + "\n")

    print()
    print("\t".join(HEADER))
    for s in results:
        print("\t".join(tsv_row(s, args)))
    print()
    print(f"明细 {args.out}")
    print(f"汇总 {tsv_path}   （列名和 h200/result.txt 一致，可直接上下拼）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
