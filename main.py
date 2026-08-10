from fastapi import FastAPI
from pydantic import BaseModel

from basic_model_serving import LLMEngine

app = FastAPI()
llm = LLMEngine()


class BatchGenerateRequest(BaseModel):
    prompts: list[str]


class BatchGenerateResponse(BaseModel):
    responses: list[str]


class BasicBatchGenerateRequest(BaseModel):
    prompt: str


class BasicBatchGenerateResponse(BaseModel):
    response: str


@app.post("/basic_generate", response_model=BasicBatchGenerateResponse)
async def generate_witout_batch(request: BasicBatchGenerateRequest):
    """
    This endpoint receives prompts in batches and generates batched responses using LLMEngine
    """
    response = llm.basic_generate_without_batch(request.prompt)
    return BasicBatchGenerateResponse(response=response)


@app.post("/generate", response_model=BatchGenerateResponse)
async def generate(request: BatchGenerateRequest):
    """
    This endpoint receives prompts in batches and generates batched responses using LLMEngine
    """
    generated_responses = llm.generate(request.prompts)
    return BatchGenerateResponse(responses=generated_responses)


"""


"""
