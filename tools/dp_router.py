"""Least-outstanding-requests HTTP router in front of independent vLLM replicas.

    .venv/bin/python tools/dp_router.py --port 8080 \
        --backend http://127.0.0.1:8000 --backend http://10.128.0.126:8000

Each request goes to the backend with the fewest requests in flight (ties to
the lowest index), and the response -- streaming or not -- is relayed as it
arrives. Round-robin is a poor fit here: a 200K prefill holds a replica for
~18 s, so two long requests in a row would land on the same one half the time.
GET /router/stats shows the per-backend counters.
"""
import argparse
import asyncio

import aiohttp
from aiohttp import web

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length",
              "host", "content-encoding"}

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--backend", action="append", required=True)
a = ap.parse_args()

inflight = {b: 0 for b in a.backend}
served = {b: 0 for b in a.backend}
lock = asyncio.Lock()


async def pick() -> str:
    async with lock:
        b = min(a.backend, key=lambda x: (inflight[x], a.backend.index(x)))
        inflight[b] += 1
        served[b] += 1
        return b


async def proxy(request: web.Request) -> web.StreamResponse:
    backend = await pick()
    try:
        body = await request.read()
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
    return web.json_response({b: {"inflight": inflight[b], "served": served[b]}
                              for b in a.backend})


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
