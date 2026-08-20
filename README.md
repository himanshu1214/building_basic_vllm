## TO START SERVER
Main server should be named as `main.py` and it should reside at the root of the project folder 

use ` uvicorn main:app --log-level debug` to start the service at `127.0.0.1:8000` 

# TO SEND THE POST REQUEST to LLM Server using batch endpoint
```
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompts": ["Explain KV cache"]
      }'
```

# TO SEND THE POST REQUEST to LLM Server using streaming endpoint
```
curl -N -X POST http://127.0.0.1:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Explain KV cache"
      }'
```

# TO RUN the Multi - serving model

```
uvicorn multi_model_server:app --host 127.0.0.1 --port 8000 --reload
```
# To test the model output in multi -model serving

```
curl http://127.0.0.1:8000/models
```