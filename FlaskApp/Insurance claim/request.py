import requests
#routing
url = 'http://localhost:5000/predict_api'
no_of_claim = int(input())
r = requests.post(url,json={'X':no_of_claim})

print(r.json())