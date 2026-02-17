import requests

ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")  # change if needed

r = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH)

print("Status:", r.status_code)
print("Studies:", r.json())
