import requests

res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "Explain AI briefly",
        "stream": False
    }
)

print(res.json()["response"])