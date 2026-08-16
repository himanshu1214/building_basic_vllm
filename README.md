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