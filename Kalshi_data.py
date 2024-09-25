# Get Market Data   
import requests

url = "https://trading-api.kalshi.com/trade-api/v2/events/INXY-24DEC31"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.json())