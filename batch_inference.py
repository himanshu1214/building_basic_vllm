from fastapi import FastAPI
from pydantic import BaseModel
from basic_model_serving import LLMEngine

app = FastAPI()
llm = LLMEngine()

@app.post("/generate", response_model=BatchGenerateResponse)
async def generate(request: BatchGenerateRequest):
    """
    This endpoint receives prompts in batches and generates batched responses using LLMEngine
    """
    generated_responses = llm.generate(request.prompts)
    return BatchGenerateResponse(responses=generated_responses)

class BatchGenerateRequest(BaseModel):
    prompts: list[str]

class BatchGenerateResponse(BaseModel):
    responses: list[str]

"""


"""