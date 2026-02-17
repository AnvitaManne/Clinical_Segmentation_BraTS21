import requests

ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# This helper finds the first study for Patient 001 to get the UUID
studies = requests.get(f"{ORTHANC}/studies", auth=AUTH).json()
PATIENT_UUID = None

for s_id in studies:
    s_info = requests.get(f"{ORTHANC}/studies/{s_id}", auth=AUTH).json()
    if s_info["PatientMainDicomTags"].get("PatientID") == "001":
        PATIENT_UUID = s_info["ParentPatient"]
        break

if PATIENT_UUID:
    patient_info = requests.get(f"{ORTHANC}/patients/{PATIENT_UUID}", auth=AUTH).json()
    studies_list = patient_info.get("Studies", [])
    print("Studies for this patient:")
    print(studies_list)
    print("Total:", len(studies_list))
else:
    print("Could not find Patient UUID.")