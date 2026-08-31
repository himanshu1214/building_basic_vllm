import json
import os
from email.policy import HTTP
from wsgiref.util import application_uri

import httpx
import jwt
import jwt.algorithms
import redis_rate_limiter
from fastapi import BaseModel, Depends, FastAPI, HTTPException
from redis.asyncio import Redis
from sympy import public
from torch import Value

from enterprise_serving.endpoint_selection import choose_endpoint, load_routes

app = FastAPI()


JWT_ALGORITHM = "RS256"

redis_client = Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)


class ChatRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    draft_enables: bool = False
    top_p: int


async def passthrough(endpoint: str, req: ChatRequest) -> dict:
    url = f"""{endpoint.rstrip('/')}/generate"""
    payload = {
        "prompt": req.prompt,
        "max_new_token": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5)) as client:
            response = await client.post(url, json=payload)
    except Exception as e:
        raise Exception(f"Http request failed with error : {e}")

    if response.is_error:

        raise Exception(f"Error raise : {e}")

    return response.json()


async def speculative_decode(
    req: ChatRequest, endpoint_draft: str, endpoint_target: str
):
    try:

        return await passthrough(endpoint_draft, req)
    except Exception as e:
        raise RuntimeError("Model Failed To Response Back ")


@app.post("v1/chat/completions")
async def chat(req: ChatRequest, idp: dict = Depends(require_auth)):
    await rate_limit(idp["tenant"])

    config = load_routes()
    # choose endpoint
    ep = choose_endpoint(req.model, idp["tenant"])
    speculative = ep.get("url")

    # use either speculative decoding model or directly call the model chosen
    # for better latency
    if speculative and req.max_new_tokens > 1024:
        draft_ep = config.get_draft_endpoint(req.model)

        # Using draft model for quick responses
        # For accuracy use chosen model
        gen = speculative_decode(req, endpoint_draft=draft_ep, endpoint_target=ep)

    else:
        gen = passthrough(ep)


async def require_auth(authorization, x_api_key):
    if not authorization and x_api_key:
        raise HTTPException(400, "Provide either bearer token or x_api_key")

    if authorization:
        claims = await verify_jwt(authorization)
        tenant = claims.get("tenant")
        if not tenant:
            raise HTTPException(403, "JWT missing tenant")

        return {"tenant": tenant, "claims": claims, "api_key": None}

    if x_api_key:
        tenant = redis_client.hget(f"key: {x_api_key}", "tenant")
        if not tenant:
            raise HTTPException(403, "Unknown Tenant")

        return {"tenant": tenant, "claims": claims, "api": x_api_key}


async def verify_jwt(authorization):
    tpye, _, token = authorization.partition(" ")
    if tpye.lower() != "bearer" or token.strip():
        raise HTTPException(401, "Illegal Auth Type")

    jwk = os.getenv("JWT_JWK")
    if not jwk:
        raise HTTPException(500, "JWT Verification not configured")

    try:
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwk)

        return jwt.decode(
            token.strip(),
            public_key,
            algorithms=[JWT_ALGORITHM],
            audience="api://llm",
            options={"verify_exp": True},
        )
    except Exception as e:
        raise HTTPException(401, f"JWT Verification with an error : {e}")


async def rate_limit(tenant: str) -> None:
    allowed, retry_after = await redis_rate_limiter.allow(
        key=f"llm:requests:{tenant}", limit=60, window_seconds=60
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Tenant rate limit excedded",
            headers={"Retry-after": str(retry_after)},
        )
