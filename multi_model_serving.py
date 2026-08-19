from pydantic import ConfigDict, BaseModel
from typing import Any
from fastapi import app
import torch
from typing import Dict
import json


class PredictionRequest(BaseModel):
    model_config = ConfigDict()
    model_id: str
    input_data: Any


@app.post("/predict")
async def predict(request: PredictionRequest):
    worker = model_manager.get_model_worker(request.model_id)
    try:
        result = worker.predict(request.input_data)
        return result
    except Exception as e:
        raise ValueError("Model failed to response ")


class ModelManager:
    def __init__(self, model_store: ModelStore, max_models: int = 2):
        self.model_store = model_store
        self.max_models = max_models
        self.model_cache = {}
        self.model_engine = ModelEngine()

    def get_model_manager(self, model_id):
        # Check the cache and
        if model_id in self.model_cache:
            self.model_cache.move_to_end(model_id)

            return self.model_engine.get_worker(model_id)

        # Other wise
        model_metadata = self.model_store.get_model(model_id)
        if not model_metadata:
            return None

        # LRU on model
        if len(self.model_cache) >= self.max_models:
            id, model_worker = self.model_cache.pop(last=False)
            self.model_worker.delete_worker(id)

        # Create new model cache
        self.model_cache[model_id] = self.model_engine.create_worker(model_metadata)
        return self.model_cache[model_id]


class Modelstore:
    def __init__(self):

        pass


class ModelEngine:
    def create_worker(self, model_metadata: ModelMetadata) -> ModelWorker:
        if model_metadata.framework == "transformers":
            self.workers[model_metadata.id] = TransformerWorker(model_metadata)
        elif model_metadata.framerwork == "torchvision":
            self.workers[model_metadata.id] = TorchVision(model_metadata)

        return self.workers[model_metadata.id]


class ModelMetadata:
    id: str
    name: str
    type: str
    framework: str
    version: str
    description: str


class ModelStore:
    def __init__(self, config_path: str):
        self.models: Dict[str, ModelMetadata] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            data = json.loads(f)
            for model in data["model"]:
                self.model[model["id"]] = ModelMetadata(**model)


class ModelWorker:
    def __init__(self):
        pass


class TransformerWorker(ModelWorker):
    def __init__(self, model_metadata):
        self.tokenizer = None
        super().__init__(model_metadata)

    def load_model(self):
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_metadata.name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_metadata.name)

    def predict(self, input_data):
        inputs = self.tokenizer(
            input_data, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad:
            output = self.model(**inputs)

        predictions = torch.softmax(output.logits, dim=1)
        return {"predictions": predictions.tolist()}


model_manager = ModelManager()
