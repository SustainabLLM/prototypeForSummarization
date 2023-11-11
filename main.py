import requests
from codecarbon import EmissionsTracker

# the API token should not be publicly availabe if I did this correctly
# I in fact did not, and it is now in a public repository :)

with open("APIToken") as f:
    token = f.read()

with open("whatToSummarize") as i:
    summarizee = i.read().replace("\n"," ")

API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-6-6"
secondUrl= "https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer"  #best this far
thirdUrl= "https://api-inference.huggingface.co/models/cnicu/t5-small-booksum"
fourthUrln= "https://api-inference.huggingface.co/models/human-centered-summarization/financial-summarization-pegasus"

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
    print("distilbart 6-6:\n"+f"{output}\n")
    print("pegasus summarizer: \n"+f"{output2}\n")
    print("t5 small booksum\n"+f"{output3}\n")
    print("pegasus financial summarization \n"+f"{output4}")
    _ = 1 + 1
finally:
    tracker.stop()
