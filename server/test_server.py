import requests
import json

request = {
    "model": "llama2-7b-chat",
    "messages": [{"role": "user", "content": "<s> [INST] Hello! How are you? [/INST]"}],
    "max_tokens": 64,
    "temperature": 0.0,
}

# Sending the PUT request
outputs = requests.put(
    url="http://127.0.0.1:5000/generate",
    data=json.dumps(request),
    headers={"Content-Type": "application/json"},
).json()[0]
print(outputs)