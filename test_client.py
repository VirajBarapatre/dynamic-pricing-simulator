import requests

url = "http://127.0.0.1:5000/predict"

# Example request body
data = {
    "unit_price": 300,
    "comp_1": 280,
    "comp_2": 310,
    "comp_3": 290,
    "holiday": 0,
    "weekend": 1,
    "month": 8
}

response = requests.post(url, json=data)

print("✅ Status Code:", response.status_code)
print("✅ Response JSON:", response.json())
