import gc
import time

import torch
from transformers import pipeline
from vllm import LLM, SamplingParams

from model_loading import create_llm

prompts = [
    "What is the meaning of life?",
    "Write a short story about a robot learning to love.",
    "Explain quantum physics in simple terms.",
    "Translate 'Hello, world!' into Spanish.",
]

llm = create_llm()
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_output_tokens=100)
start_time = time.time()
vllm_output = llm.generate(prompts, sampling_params=sampling_params)
vllm_time = time.time() - start_time

print("vLLM Output:", vllm_output)
print("Total time taken by vLLM:", vllm_time, "seconds")
