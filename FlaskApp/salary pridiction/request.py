import requests
#routing
url = 'http://localhost:5000/predict_api'
exp = int(input())
r = requests.post(url,json={'YearsExperience':exp})

print(r.json())