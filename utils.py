import gc
import time

import torch


def clean_gpu(model):
    if model:
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()


clean_gpu(None)
