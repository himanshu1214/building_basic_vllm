import json
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
import tritonclient.http as httpclient
from fastapi import app
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
        self.model_cache = OrderedDict()
        self.model_engine = ModelEngine()

    def get_model_worker(self, model_id):
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
            id, model_worker = self.model_cache.popitem(last=False)
            self.model_worker.delete_worker(id)

        # Create new model cache
        self.model_cache[model_id] = self.model_engine.create_worker(model_metadata)
        return self.model_cache[model_id]

    def list_loaded_models(self):
        return {
            model_id: worker.model_metdata.name
            for model_id, worker in self.model_cache.items()
        }


class ModelEngine:
    """
    This class is used to create/get/delete the model worker based on the model
    """

    def __init__(self):
        self.workers = {}

    def create_worker(self, model_metadata: ModelMetadata) -> ModelWorker:
        if model_metadata.framework == "transformers":
            self.workers[model_metadata.id] = TransformerWorker(model_metadata)
        elif model_metadata.framerwork == "torchvision":
            self.workers[model_metadata.id] = TorchVisionWorker(model_metadata)

        return self.workers[model_metadata.id]

    def get_worker(self, model_id: str) -> ModelWorker:
        """this method is used to get the worker based on the model_id"""
        return self.workers[model_id]

    def delete_worker(self, model_id) -> ModelWorker:
        """
        This is used to delete worker based on model_id
        """
        if model_id in self.workers:
            del self.workers[model_id]


class ModelMetadata:
    id: str
    name: str
    type: str
    framework: str
    version: str
    description: str


class ModelStore:
    """
    This class store the model metadata and get the model based on Model Metadata
    """

    def __init__(self, config_path: str):
        self.models: Dict[str, ModelMetadata] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            data = json.loads(f)
            for model in data["model"]:
                self.model[model["id"]] = ModelMetadata(**model)

    def get_model(self, model_id):
        return self.model[model_id]

    def list_models(self):
        return self.models


class ModelWorker(ABC):
    def __init__(self, model_metadata):
        self.model_metadata = model_metadata
        self.mode = None
        self._load_model()

    def _load_model(self):
        pass

    def predict(self):
        pass


class TransformerWorker(ModelWorker):
    """
    This class uses the Model Worker and implements the tokenization and
    predictions
    """

    def __init__(self, model_metadata):
        self.tokenizer = None
        super().__init__(model_metadata)

    def load_model(self):
        """
        Model load and tokenizer
        """
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_metadata.name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_metadata.name)

    def predict(self, input_data):
        """
        Main method for prediction
        """
        inputs = self.tokenizer(
            input_data, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            output = self.model(**inputs)

        predictions = torch.softmax(output.logits, dim=1)
        return {"predictions": predictions.tolist()}


class TorchVisionWorker:
    """
    Torch Vision Worker
    """

    def __init__(self, model_metadata):
        self.transform = None
        super.__init__(model_metadata)

    def _load_model(self):
        """
        Load models
        """
        if self.model is None:
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.model.eval()
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def predict(self, input_data):
        """
        Prediction for model using TorchVision framework
        """
        if self.model is None or self.transform is None:
            raise (f"Model not loaded : {self.mode} or transform : {self.transform}")

        if isinstance(input_data, str):
            image = Image.open(input_data).convert("RGB")
        else:
            image = input_data

        image_tnsor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tnsor)
        predictions = torch.softmax(output, dim=1)
        return {"predictions": predictions.tolist()}


class TritonWorker(ModelWorker):
    """
    Triton worker class
    """

    def __init__(self, model_metadata):
        self.host = "0.0.0.0:8009"
        self.client = httpclient.InferenceServerClient(url=self.host)
        super().__init__(model_metadata)

    def _load_model(self):
        url = f"http://{self.host}/v2/repository/models/{self.model.metadata.name}/load"
        response = requests.post(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Model failed with response code: {response.status_code}"
            )
        if self.client.is_model_ready(self.model.metadata.name):
            raise RuntimeError(f"Failed to load the model")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = []
        for name, data in input_data.items():
            if not isinstance(data, np.ndarray):
                try:
                    data_shape = data["shape"]
                    content = data["data"]
                    array = np.array(content, dtype=np.float32).reshape(data_shape)
                except:
                    raise ValueError("Some issue with the data")
            else:
                array = data.astype(np.float32)

            input_tensor = httpclient.InferInput(name, array.shape, "FP32")
            input_tensor.set_data_from_numpy(array)
            inputs.append(input_tensor)

        output_name = "fc6_1"
        response = self.client.infer(
            model_name=self.model_metadata.name,
            inputs=inputs,
            outputs=[httpclient.InferRequestedOutput(output_name)],
        )

        predictions = {output_name: response.as_numpy(output_name).tolist()}
        return predictions

    def __del__(self):
        try:
            unload_url = f"http://{self.host}/v2/repository/models/{self.model_metadata.name}/unload"
            requests.post(unload_url)
        except:
            pass


model_manager = ModelManager()
