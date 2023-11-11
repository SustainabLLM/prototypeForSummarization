import requests
from codecarbon import EmissionsTracker

# the API token should not be publicly availabe if I did this correctly
# I in fact did not, and it is now in a public repository :)

with open("APIToken") as f:
    token = f.read()

with open("whatToSummarize") as i:
    summarizee = i.read()

API_URL = "https://api-inference.huggingface.co/models/google/pegasus-xsum"
secondUrl= "https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer"
thirdUrl= "https://api-inference.huggingface.co/models/google/bigbird-pegasus-large-bigpatent"
fourthUrln= "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
headers = {"Authorization": f"Bearer {token}"}

tracker = EmissionsTracker()
tracker.start()
try:
    def query(payload,api):
        response = requests.post(api, headers=headers, json=payload)
        return response.json()


    output = query({"inputs": f"{summarizee}"},API_URL)
    output2=query({"inputs": f"{summarizee}"},secondUrl)
    output3=query({"inputs": f"{summarizee}"},thirdUrl)
    output4=query({"inputs": f"{summarizee}"},fourthUrln)
    print(output)
    print(output2)
    print(output3)
    print(output4)
    _ = 1 + 1
finally:
    tracker.stop()
