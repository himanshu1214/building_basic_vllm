import asyncio
import collections
import json
import logging
import multiprocessing as mp
import sys
import threading
import uuid
from queue import Empty, Queue
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Sequence:
    def __init__(
        self,
        id: str,
        prompt: str,
        loop=None,
        client_stream=None,
    ):
        self.id = id
        self.output = []
        self.prompt = prompt
        self.isFinished = False
        self.token_count = 0
        self.loop = loop
        self.client_stream = client_stream


class LLMEngine:
    """
    This class is actually responsible for interfacing with the model and generating responses
    """

    def __init__(self):
        self.workload_manager = WorkloadManager()
        self.model_executor = ModelExecutor()
        self.max_tokens = 20
        self.model_executor.setup_workers("facebook/opt-125m")
        self.thread = threading.Thread(target=self.request_processing_loop, daemon=True)
        self.thread.start()

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

    def generate(self, prompts: List[str]) -> List:
        request_ids = []
        for prompt in prompts:
            prompt_id = self.workload_manager.add_request_to_queue(prompt)
            request_ids.append(prompt_id)

        # Keep generating responses for the sequences untill all the current prompts are responded
        # We are keeping track of the sequences using workload manager
        while not self._isbatch_finished(request_ids):
            sequences: List[Sequence] = self.workload_manager.generate_batched_request()
            response = self.model_executor.execute_batch(sequences)

            for res in response[1]:  # 1st index is 'completed'
                self.workload_manager.remove_active_sequence(res["prompt_id"])
                self.workload_manager.update_sequence_output(
                    res["prompt_id"], res["generated_response"], isFinished=True
                )

        ###

        generated_texts = []
        for request_id in request_ids:
            generated_text = self.workload_manager.get_generated_text(
                request_id
            ).output[0]
            generated_texts.append(generated_text)
            self.workload_manager.remove_finished_sequence(request_id)
        return generated_texts

    def basic_generate_without_batch(self, prompt):
        sequence = Sequence(str(uuid.uuid4()), prompt, None, None)
        results = self.model_executor.execute_batch([sequence])
        return results[1][0][
            "generated_response"
        ]  # 1st index is 'completed' and 0th index is the first sequence

    def request_processing_loop(self):
        while True:
            try:
                active_sequences = self.workload_manager.generate_batched_request(
                    is_streaming=True
                )
                prompts = [
                    {"prompt": seq.prompt, "request_id": seq.id}
                    for seq in active_sequences
                ]

                tokens_result = self.model_executor.execute_forward_batch(prompts)

                for result in tokens_result:
                    seq = self.workload_manager.get_sequence(result["request_id"])
                    if result["is_finished"] or seq.token_count > self.max_tokens:
                        asyncio.run_coroutine_threadsafe(
                            seq.client_stream.put(None), seq.loop
                        )
                        seq.isFinished = True
                        self.workload_manager.remove_finished_sequence(
                            result["request_id"]
                        )
                    else:
                        asyncio.run_coroutine_threadsafe(
                            seq.client_stream.put(
                                json.dumps(
                                    {
                                        "token": result["token"],
                                        "sequence_id": result["request_id"],
                                    }
                                )
                            ),
                            seq.loop,
                        )
                        self.workload_manager.update_sequence_output(
                            result["request_id"], result["token"], result["is_finished"]
                        )
            except Exception as e:
                print(f"Error found while processing request loop with : {e}")

    async def event_generator(self, loop: asyncio.AbstractEventLoop, prompt: str):

        asyncio.set_event_loop(loop)
        queue = asyncio.Queue()  # client loop

        seq_id = self.workload_manager.add_streaming_request(prompt, queue, loop)
        print(
            f"Created queue for sequence {seq_id} in loop {id(loop)} and queue {id(queue._get_loop())}"
        )  # Debug print
        try:
            while True:
                data = await queue.get()
                if data is None:
                    break
                yield f"data: {data}\n\n\ "
        except Exception as e:
            print(f"Error in sequence {seq_id} : {e}")
        finally:
            self.workload_manager.remove_finished_sequence(seq_id)
            print("Removed the sequence from seq_map, stream_map")


class ModelExecutor:
    """
    This class is responsible for setting up the model workers and workers group
    """

    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker_process = None
        logger.debug("ModelExecutor initialized with queues")

    def setup_workers(self, model_name):
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
        )

        self.worker_process.start()
        print(
            "Worker PID:",
            self.worker_process.pid,
            "alive:",
            self.worker_process.is_alive(),
            flush=True,
        )

    def execute_batch(self, prompts: List[Sequence]) -> tuple[str, Dict[str, str]]:
        print(
            "Before queue put:",
            "alive =",
            self.worker_process.is_alive(),
            "exitcode =",
            self.worker_process.exitcode,
            flush=True,
        )
        self.task_queue.put((prompts, False))
        try:
            results = self.result_queue.get(timeout=120)
            return results
        except Empty:
            print(
                "After timeout:",
                "alive =",
                self.worker_process.is_alive(),
                "exitcode =",
                self.worker_process.exitcode,
                flush=True,
            )
            raise RuntimeError(
                "Model worker did not return a result. "
                "Check whether the worker process crashed."
            )

    def execute_forward_batch(self, prompts):

        if not prompts:
            logger.debug("No prompts received")

            return []

        logger.debug(f"Adding streaming batch prompts into task queue: {prompts}")

        # send the batch with streaming flag
        self.task_queue.put((prompts, True))

        logger.debug(f"Awaiting results back from result queue")
        result_type, result = self.result_queue.get()

        logger.debug(f"RESPONSE received : {result}")

        if result_type == "stream":
            return result
        else:
            raise ValueError("Unrecognized result format")


class ModelWorker:
    """
    This class is the one interfacing with the model and decoding the generated output
    """

    def __init__(self, model_name):
        print("Starting ModelWorker...", flush=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device, flush=True)

        print("Loading model...", flush=True)
        self.model, self.tokenizer = ModelManager.load_model(model_name)

        print("Moving model to GPU...", flush=True)
        self.model.to(self.device)
        self.model.eval()

        print("Model ready", flush=True)
        self.max_tokens = 20

    @staticmethod
    def run(model_name, task_queue: Queue, result_queue: Queue):

        # put the request into the task queue
        try:
            print("Worker process starting...", flush=True)
            worker = ModelWorker(model_name)
            print(f"Worker started on: {worker.device}", flush=True)
            while True:
                sequences, is_streaming = (
                    task_queue.get()
                )  # Tuple[List[Sequence], bool]
                if not sequences:
                    logger.debug("No sequences found")
                    break
                if is_streaming:
                    result_queue.put(
                        ("stream", worker.generate_forward_batch(sequences))
                    )
                else:
                    result_queue.put(("completed", worker.generate(sequences)))

        except Exception as e:
            import traceback

            traceback.print_exc()
            result_queue.put(("error", repr(e)))

    def generate(self, prompts: List[Sequence]) -> Dict[str, str]:
        """ "
        Initialize the model to generate the response and
        the generated is decoded back to the text format using tokenizer
        """
        prompt_txt = [p.prompt for p in prompts]
        request_ids = [p.id for p in prompts]

        inputs = self.tokenizer(
            prompt_txt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_tokens,
                attention_mask=inputs.attention_mask,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.batch_decode(outputs)
        return [
            {
                "prompt_id": request_id,
                "generated_response": generated_text,
            }
            for request_id, generated_text in zip(request_ids, generated_text)
        ]

    def generate_forward_batch(self, prompts: List[Dict[str, Any]]):
        """
        Use this method on streaming prompt sequences
        """
        logger.debug("Begin generating stream token")
        # add optional pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # tokenize
        encoded_prompts = self.tokenizer(
            [p["prompt"] for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        logger.debug("Added the prompt sequences list as encoded prompts")

        with torch.no_grad():
            model_response = self.model(
                input_ids=encoded_prompts.input_ids,
                attention_mask=encoded_prompts.attention_mask,
                use_cache=False,
            )

            next_token_logit = model_response.logits[:, -1, :]
            next_token = torch.multinomial(
                torch.softmax(next_token_logit / 0.7, dim=-1), num_samples=1
            ).squeeze(-1)

            results = []
            for i, prompt_data in enumerate(prompts):
                token = self.tokenizer.decode(
                    next_token[i].unsqueeze(0), skip_special_tokens=True
                )
                logger.debug(
                    f"Generate token for the given prompt '{prompt_data['prompt']}': '{token}' "
                )
                results.append(
                    {
                        "request_id": prompt_data["request_id"],
                        "token": token,
                        "is_finished": token == self.tokenizer.eos_token,
                    }
                )
            return results


class ModelManager:
    """
    This class is responsible for loading the model
    """

    @staticmethod
    def load_model(model_name="facebook/opt-125m"):
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
        self.request_map = {}
        self.active_requests = []
        self.incoming_requests = Queue()
        self.incoming_streaming_requests = Queue()
        self.seq_map: Dict[str:Sequence] = {}
        self.active_streaming_sequence: List[Sequence] = []

    def add_request_to_queue(self, prompt: str) -> str:
        # Add the request to the queue
        request_id = str(uuid.uuid4())
        sequence = Sequence(request_id, prompt, None, None)
        self.incoming_requests.put(sequence)
        self.request_map[request_id] = sequence
        return request_id

    def get_generated_text(self, request_id: str) -> Sequence:
        return self.request_map[request_id]

    def get_sequence(self, seq_id: str) -> List[Sequence]:
        return self.seq_map[seq_id]

    def add_streaming_request(self, prompt: str, client_stream: Queue, loop):
        """
        Add the prompt sequence into  the request queue and to the sequence map for tracking
        """
        request_id = str(uuid.uuid4())
        sequence = Sequence(request_id, prompt, client_stream, loop)
        self.incoming_streaming_requests.put(
            sequence
        )  # Add Sequence into the request Queue
        self.seq_map[request_id] = sequence  # Tracking the prompt here
        return request_id

    def generate_batched_request(self, is_streaming: bool = False) -> List[Sequence]:
        if is_streaming:
            while (
                len(self.active_streaming_sequence) < self.batch_size
                and not self.incoming_streaming_requests.empty()
            ):
                sequence = self.incoming_streaming_requests.get()
                self.active_streaming_sequence.append(sequence)
            return self.active_streaming_sequence
        else:
            while (
                len(self.active_requests) < self.batch_size
                and not self.incoming_requests.empty()
            ):
                sequence = self.incoming_requests.get()
                self.active_requests.append(sequence)

        return self.active_requests

    def update_sequence_output(
        self, sequence_id: str, token: str, isFinished: bool
    ) -> Sequence | None:
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

    def remove_active_sequence(self, sequence_id: str) -> None:
        """
        This method is responsible for removing the sequence based on the request_id/seq_id from
        """
        if sequence_id in self.request_map:
            sequence = self.request_map[sequence_id]
            if sequence in self.active_requests:
                self.active_requests.remove(sequence)
            if sequence in self.active_streaming_sequence:
                self.active_streaming_sequence.remove(sequence)

    def is_sequence_finished(self, sequence_id: str) -> bool:
        """
        Check if the sequence is finished based on the
        """
        if sequence_id in self.request_map:
            sequence = self.request_map[sequence_id]
            return sequence.isFinished
        return False

    def remove_finished_sequence(self, seq_id):
        if seq_id in self.seq_map:
            sequence = self.seq_map[seq_id]
            if sequence in self.active_requests:
                self.active_requests.remove(sequence)
            if sequence in self.active_streaming_sequence:
                self.active_streaming_sequence.remove(sequence)
            del self.seq_map[seq_id]
