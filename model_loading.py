import time
from vllm import LLM, SamplingParams

def create_llm():
    llm = LLM(model="Qwen/Qwen2.5-0.5B", 
                dtype="float16",
                trust_remote_code=True,
                max_model_len=2048,
                gpu_memory_utilization=0.80,)
    return llm