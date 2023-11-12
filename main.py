import requests
from codecarbon import EmissionsTracker
import scipper

# the API token should not be publicly availabe if I did this correctly
# I in fact did not, and it is now in a public repository :)
kysymys=input("What do you wanna know? \n")



with open("APIToken") as f:
    token = f.read()

with open("whatToSummarize","r",encoding='utf-8') as i:
    summarizee = i.read().replace("\n"," ")

API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-6-6" #even better tho
headers = {"Authorization": f"Bearer {token}"}

def query(payload,api):
        response = requests.post(api, headers=headers, json=payload)
        return response.json()


output = query({"inputs": f"{scipper.inference(kysymys)}"},API_URL)

print("distilbart 6-6:\n"+f"{output}\n")
