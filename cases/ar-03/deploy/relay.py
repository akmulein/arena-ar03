"""Fixed upstream to the operator API; never a general-purpose HTTP proxy."""

import os

import httpx
from fastapi import FastAPI, Request, Response

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
UPSTREAM = os.environ.get("AR03_OPERATOR_URL", "http://host.docker.internal:8799")


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def relay(path: str, request: Request):
    if path not in {"healthz", "status", "propose", "evaluate", "finalize", "progress"} and not (
        path.startswith("result/") or path.startswith("progress/")
    ):
        return Response(status_code=404)
    body = await request.body()
    if len(body) > 72 * 1024 * 1024:
        return Response(status_code=413)
    async with httpx.AsyncClient(trust_env=False, timeout=600, follow_redirects=False) as client:
        result = await client.request(request.method, UPSTREAM + "/" + path,
            content=body, headers={"Authorization": request.headers.get("Authorization", ""),
                                  "Content-Type": "application/json"})
    return Response(result.content, status_code=result.status_code,
                    media_type=result.headers.get("Content-Type", "application/json"))
