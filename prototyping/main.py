import requests

#the API token should not be publicly availabe if I did this correctly
with open("APIToken") as f:
    token = f.read()

API_URL = "https://api-inference.huggingface.co/models/google/pegasus-xsum"
headers = {"Authorization": f"Bearer {token}"}

from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
try:
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
     "inputs": "Stainless steel production is a complex process that involves various considerations to ensure the quality, efficiency, and sustainability of the final product. Here are some common things to take into account in stainless steel production"
     })
    print(output)
    _ = 1 + 1
finally:
    tracker.stop()
