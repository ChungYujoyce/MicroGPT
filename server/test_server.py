import requests
import json

request = {
    "prompts": ["Hello! How are you?"],
    "tokens_to_generate": 64,
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.0,
    "stop_words_list": [],
}

# Sending the PUT request
outputs = requests.put(
    url="http://127.0.0.1:5000/generate",
    data=json.dumps(request),
    headers={"Content-Type": "application/json"},
).json()[0]
print(outputs)