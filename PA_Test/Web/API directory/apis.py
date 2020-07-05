import requests

response = requests.get("https://api.twitch.tv/thefirejedi/channel")
print(response.status_code)
print(response)
