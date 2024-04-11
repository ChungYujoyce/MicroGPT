import requests
import json

request = {
    "model": "llama2-7b-chat",
    "prompt": ["<s> [INST] Hello! How are you? [/INST]"],
    "max_tokens": 128,
    "temperature": 0.0,
    "stop": ["."]
}

# Sending the PUT request
outputs = requests.post(
    url="http://172.18.0.2:5000/v1/completions",
    data=json.dumps(request),
    headers={"Content-Type": "application/json"},
).json()
print(outputs)