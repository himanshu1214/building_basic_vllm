# Host one model with Ray Serve

`single_model_hosting.py` runs one vLLM engine in one Ray Serve replica. vLLM
handles continuous batching inside that replica, so concurrent HTTP requests do
not require loading additional copies of the model.

## Start the service

From the repository root, use the existing environment:

```bash
venv3/bin/serve run enterprise_serving.single_model_hosting:app
```

Ray starts a local cluster when one is not already running and exposes the API
at `http://127.0.0.1:8000`. Do not start Uvicorn separately for this service.

The default model is `Qwen/Qwen2.5-0.5B`. Override it with a Hugging Face model
ID or a local model directory:

```bash
MODEL_ID=/models/my-model \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.9 \
venv3/bin/serve run enterprise_serving.single_model_hosting:app
```

For tensor parallel inference, reserve the same number of GPUs as the tensor
parallel size:

```bash
MODEL_ID=/models/my-model \
TENSOR_PARALLEL_SIZE=2 \
venv3/bin/serve run enterprise_serving.single_model_hosting:app
```

Supported settings are `MODEL_ID`, `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, and `TRUST_REMOTE_CODE`.

## Call the model

Check readiness after the model has loaded:

```bash
curl http://127.0.0.1:8000/health
```

Generate text:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain KV cache in two sentences.",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

The request also accepts `text` as an alias for `prompt`, preserving the field
name used by the original prototype.

## Route a selected model to this endpoint

`endpoint_selection.py` loads `routes.yaml` through `load_routes()`. The file
maps a public model name to the base URL of a Ray Serve application. For a
different deployment environment, set `MODEL_ROUTES_PATH` to another YAML file.

```python
endpoint = choose_endpoint("qwen-small", tenant="premium")
# endpoint is then forwarded to: f"{endpoint}/generate"
```

The configuration supports aliases, per-tenant endpoint overrides, and a
weighted canary endpoint. It is re-read for each selection, so configuration
updates apply without restarting the gateway.

Stop the foreground `serve run` process with Ctrl-C. Ray Serve then removes the
application and releases the model resources.
