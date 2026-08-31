"""Serve one vLLM model behind a Ray Serve HTTP endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field
from ray import serve
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

logger = logging.getLogger("ray.serve")

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
api = FastAPI(title="Ray Serve single-model vLLM API", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: Annotated[
        str,
        Field(
            min_length=1,
            validation_alias=AliasChoices("prompt", "text"),
            description="Prompt to send to the model. The legacy key 'text' is also accepted.",
        ),
    ]
    max_new_tokens: Annotated[int, Field(ge=1, le=4096)] = 256
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95
    stop: str | list[str] | None = None
    seed: int | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    request_id: str
    model: str
    text: str
    finish_reason: str | None
    usage: Usage


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _fraction_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1, got {value}")
    return value


def _bool_from_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}")


@serve.deployment(
    name="single_model_vllm",
    num_replicas=1,
    max_ongoing_requests=64,
)
@serve.ingress(api)
class SingleModelVLLM:
    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
    ) -> None:
        self.model_id = model_id
        engine_args = AsyncEngineArgs(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @api.get("/health")
    async def health(self) -> dict[str, str]:
        return {"status": "ok", "model": self.model_id}

    @api.post("/generate", response_model=GenerateResponse)
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            seed=request.seed,
        )

        final_output = None
        try:
            async for output in self.engine.generate(
                request.prompt,
                sampling_params,
                request_id,
            ):
                # vLLM returns cumulative text by default, so only the final output
                # should be returned rather than concatenating every iteration.
                final_output = output
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise
        except ValueError as exc:
            await self.engine.abort(request_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            await self.engine.abort(request_id)
            logger.exception("Generation failed for request %s", request_id)
            raise HTTPException(
                status_code=500, detail="Model generation failed"
            ) from exc

        if final_output is None or not final_output.outputs:
            raise HTTPException(status_code=500, detail="Model returned no output")

        completion = final_output.outputs[0]
        prompt_tokens = len(final_output.prompt_token_ids or [])
        completion_tokens = len(completion.token_ids)
        return GenerateResponse(
            request_id=request_id,
            model=self.model_id,
            text=completion.text,
            finish_reason=completion.finish_reason,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


def build_app():
    """Build the Serve application from environment-based model settings."""
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    tensor_parallel_size = _positive_int_from_env("TENSOR_PARALLEL_SIZE", 1)
    max_model_len = _positive_int_from_env("MAX_MODEL_LEN", 2048)
    gpu_memory_utilization = _fraction_from_env("GPU_MEMORY_UTILIZATION", 0.8)
    trust_remote_code = _bool_from_env("TRUST_REMOTE_CODE", False)

    return SingleModelVLLM.options(
        ray_actor_options={
            "num_cpus": 1,
            "num_gpus": tensor_parallel_size,
        }
    ).bind(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )


# The Ray Serve CLI imports this application object.
app = build_app()
