from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from multi_model_serving import ModelManager, ModelStore

app = FastAPI()
model_store = ModelStore("model_config.json")
model_manager = ModelManager(model_store)


class PredictionRequest(BaseModel):
    model_config: ConfigDict(protected_namespaces=())
    model_id: str
    input_data: Any


@app.post("/predict")
async def predict(request: PredictionRequest):
    worker = model_manager.get_model_worker(request.model_id)

    try:
        response = worker.predict(request.input_data)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models")
async def list_models():
    return {
        "available_models": model_store.list_models(),
        "loaded_models": model_manager.list_loaded_models(),
    }
