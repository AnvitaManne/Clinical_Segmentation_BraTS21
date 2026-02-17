import requests
import json

ORTHANC = "http://localhost:8042"
# Note: This UUID is the internal Orthanc ID for the patient
PATIENT_UUID = "e193a01e-cf8d30ad-0affefd3-32ce934e-32ffce72"

# Get patient info
patient_info = requests.get(f"{ORTHANC}/patients/{PATIENT_UUID}").json()

studies = patient_info.get("Studies", [])

print("Studies for this patient:")
print(studies)
print("Total:", len(studies))

