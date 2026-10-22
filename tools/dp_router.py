"""Prefix-affine HTTP router in front of independent vLLM replicas.

    .venv/bin/python tools/dp_router.py --port 8080 \
        --backend http://127.0.0.1:8000 --backend http://10.128.0.126:8000

Each replica has its own prefix cache, so the turns of one conversation (an
agent resends its whole history every turn) must keep landing on the same
replica. A request is keyed by what stays fixed across a conversation's turns
-- the system messages plus the first user message for chat, the head of the
prompt for completions, or an ``X-Session-Id`` header -- and the key picks a
replica by rendezvous hashing. If that replica has more than ``--spill``
requests in flight beyond the least-loaded one, the request goes to the
least-loaded replica instead (fewest in flight, ties to the lowest index);
requests without a key (GETs, bodies that are not JSON) always do. Whole
prompts are not used as the key: that would change every turn, and a long
system prompt shared by every task would send every task to one replica.

The response -- streaming or not -- is relayed as it arrives. GET
/router/stats shows the per-backend counters.
"""
import argparse
import asyncio
import hashlib
import json

import aiohttp
from aiohttp import web

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length",
              "host", "content-encoding"}

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--backend", action="append", required=True)
ap.add_argument("--spill", type=int, default=4,
                help="in-flight excess over the least-loaded replica at which "
                     "a keyed request leaves its affine replica")
a = ap.parse_args()

inflight = {b: 0 for b in a.backend}
served = {b: 0 for b in a.backend}
routed = {"affine": 0, "spilled": 0, "unkeyed": 0}
lock = asyncio.Lock()

COMPLETION_KEY_CHARS = 32768
COMPLETION_KEY_TOKENS = 8192


def _text(content) -> str:
    """Chat message content (a string or a list of parts) as text."""
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def affinity_key(request: web.Request, body: bytes) -> str | None:
    session = request.headers.get("X-Session-Id")
    if session:
        return "session:" + session
    if request.method != "POST" or not body:
        return None
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(req, dict):
        return None
    messages = req.get("messages")
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                return None
            parts.append(f"{m.get('role')}\0{_text(m.get('content'))}")
            if m.get("role") == "user":
                break
        return "chat:" + "\1".join(parts)
    prompt = req.get("prompt")
    if isinstance(prompt, str):
        return "text:" + prompt[:COMPLETION_KEY_CHARS]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return "ids:" + ",".join(map(str, prompt[:COMPLETION_KEY_TOKENS]))
    return None


def _rendezvous(key: str) -> str:
    return max(a.backend, key=lambda b: hashlib.blake2b(
        f"{b}|{key}".encode(), digest_size=8).digest())


async def pick(key: str | None) -> str:
    async with lock:
        least = min(a.backend, key=lambda x: (inflight[x], a.backend.index(x)))
        if key is None:
            b = least
            routed["unkeyed"] += 1
        else:
            b = _rendezvous(key)
            if inflight[b] - inflight[least] > a.spill:
                b = least
                routed["spilled"] += 1
            else:
                routed["affine"] += 1
        inflight[b] += 1
        served[b] += 1
        return b


async def proxy(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    backend = await pick(affinity_key(request, body))
    try:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        async with request.app["session"].request(
                request.method, backend + request.rel_url.path_qs,
                data=body, headers=headers) as upstream:
            resp = web.StreamResponse(status=upstream.status, headers={
                k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP})
            await resp.prepare(request)
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    finally:
        async with lock:
            inflight[backend] -= 1


async def stats(_: web.Request) -> web.Response:
    out = {b: {"inflight": inflight[b], "served": served[b]} for b in a.backend}
    out["routing"] = dict(routed)
    return web.json_response(out)


async def health(_: web.Request) -> web.Response:
    s = app["session"]
    ok = []
    for b in a.backend:
        try:
            async with s.get(b + "/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                ok.append(r.status == 200)
        except Exception:  # noqa: BLE001
            ok.append(False)
    return web.json_response(dict(zip(a.backend, ok)), status=200 if all(ok) else 503)


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))


async def on_cleanup(app):
    await app["session"].close()


app = web.Application(client_max_size=1 << 30)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
app.router.add_get("/router/stats", stats)
app.router.add_get("/health", health)
app.router.add_route("*", "/{tail:.*}", proxy)
web.run_app(app, port=a.port, access_log=None)
