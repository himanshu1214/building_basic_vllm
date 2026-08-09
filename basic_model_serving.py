import collections
import multiprocessing as mp

import torch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
import uuid
from typing import Dict, List
from queue import Queue

class LLMEngine:
    """
    This class is actually responsible for interfacing with the model and generating responses
    """
    def __init__(self):
        self.workload_manager = WorkloadManager()
        self.model_executor = ModelExecutor()
        self.max_tokens = 20

    def _isbatch_finished(self, request_ids):
        """
        Check the request_id / sequence_id to see if the sequence is finished or not
        """
        for request_id in request_ids:
            if self.workload_manager.is_sequence_finished(request_id):
                continue
            else:
                return False
        return True

    def generate(self, prompts):
        responses = []
        request_ids = []
        for prompt in prompts:
            prompt_id = self.workload_manager.add_request_to_queue(prompt)
            request_ids.append(prompt_id)

        # Keep generating responses for the sequences untill all the current prompts are responded
        # We are keeping track of the sequences using workload manager
        while not self._isbatch_finished(request_ids):
            sequences = self.workload_manager.generate_batched_request()
            response = self.model_executor.execute_batch(sequences)
            responses.append(response[0]['generated_response'])
        for res in response:
            self.workload_manager.update_sequence_output(res, )

        ### 
        
        generated_texts = []
        for request_id in request_ids:
            generated_text = self.workload_manager.get_generated_text(request_id).output[0]
            generated_texts.append(generated_text)
            self.workload_manager.remove_request(request_id)
        return generated_texts


class ModelExecutor:
    """
    This class is responsible for setting up the model workers and workers group
    """
    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def setup_workers(self, model_name):
        self.worker_process = mp.Process(target=ModelWorker.run,
                                         args=(model_name, self.task_queue, self.result_queue))

        self.worker_process.start()

    def execute_batch(self, prompt):
        self.task_queue.put((prompt, False))
        results = self.result_queue.get()
        return results
    
class ModelWorker:
    """
    This class is the one interfacing with the model and decoding the generated output
    """
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = ModelManager.load_model(model_name)
    @staticmethod
    def run(model_name, task_queue, result_queue):
        worker = ModelWorker(model_name)
        # put the request into the task queue
        while True:
            request = task_queue.get()
            result_queue.put(('completed', worker.generate(request)))

    def generate(self,prompt) -> Dict[str, str]:
        """"
        Initialize the model to generate the response and 
        the generated is decoded back to the text format using tokenizer
        """
        outputs = self.model.generate()
        generated_text = self.tokenizer.decode(outputs[0])
        return {
            'prompt_id': prompt.id,
            'generated_response': generated_text,
        }

class ModelManager:
    """
    This class is responsible for loading the model 
    """
    @staticmethod
    def load_model(model_name= 'facebook/opt-125m'):
        # Load the model and tokenizer here
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

class WorkloadManager:
    """
    Thi class is responible for managing the request order, tracking the request and response
    """
    def __init__(self):
        self.batch_size = 4
        self.request_map = collections.defaultdict(str)
        self.active_requests = []
        self.incoming_requests = Queue()


    def add_request_to_queue(self, prompt):
        # Add the request to the queue
        request_id = str(uuid.uuid4())
        sequence = Sequence(request_id, prompt, None, None)
        self.incoming_requests.put(sequence)    
        self.request_map[request_id] = sequence
        return request_id

    def get_generated_text(self, request_id):
        return self.request_map[request_id]
    
    def generate_batched_request(self) -> List[Sequence]:
        while len(self.active_requests) < self.batch_size and not self.incoming_requests.empty():
            sequence = self.incoming_requests.get()
            self.active_requests.append(sequence)

        return self.active_requests

    def update_sequence_output(self, sequence_id, token, isFinished)-> Sequence | None:
        """
        Get the sequence to update with the generated token and mark it as finished as appropriate
        """
        if sequence_id in self.request_map:
            sequence = self.request_map[sequence_id]
            sequence.output.append(token)
            sequence.prompt += token
            sequence.token_count += 1
            sequence.isFinished = isFinished
            return sequence
        return None

    def is_sequence_finished(self, sequence_id):
        """
        Check if the sequence is finished based on the 
        """
        if sequence_id in self.request_map:
            sequence = self.request_map[sequence_id]
            return sequence.isFinished
        return False

class Sequence:
    def __init__(self, id, prompt, response, timestamp):
        self.id = id
        self.output = []
        self.prompt = prompt
        self.response = response
        self.timestamp = timestamp
        self.isFinished = False