from fastapi import FastAPI, Depends
from pydantic import BaseModel

from basic_model_serving import LLMEngine
from model_loading import create_llm
from vllm import SamplingParams
import multiprocessing
import atexit

# Create FastAPI app
app = FastAPI()

# Create LLM instance
_llm = None
_llm_lock = multiprocessing.Lock()

def cleanup():
    global _llm
    if _llm is not None:
        try:
            _llm._cleanup()
        except:
            pass
        _llm = None

def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = LLMEngine()
            # Register cleanup
            atexit.register(cleanup)
        return _llm
    
class BatchGenerateRequest(BaseModel):
    prompts: list[str]


class BatchGenerateResponse(BaseModel):
    responses: list[str]


class BasicBatchGenerateRequest(BaseModel):
    prompt: str


class BasicBatchGenerateResponse(BaseModel):
    response: str


@app.post("/basic_generate", response_model=BasicBatchGenerateResponse)
async def generate_witout_batch(request: BasicBatchGenerateRequest,  llm: LLMEngine = Depends(get_llm)):
    """
    This endpoint receives prompts in batches and generates batched responses using LLMEngine
    """
    response = llm.basic_generate_without_batch(request.prompt)
    return BasicBatchGenerateResponse(response=response)


@app.post("/generate", response_model=BatchGenerateResponse)
async def generate(request: BatchGenerateRequest,  llm: LLMEngine = Depends(get_llm)):
    """
    This endpoint receives prompts in batches and generates batched responses using LLMEngine
    """

    generated_responses = llm.generate(request.prompts)
    return BatchGenerateResponse(responses=generated_responses)


def signal_handler(signum, frame):
    cleanup()
    exit(0)

