import requests

ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# Change this ID to match the patient you uploaded via 3D Slicer
PATIENT_ID = "001"  

studies = requests.get(
    f"{ORTHANC_URL}/studies",
    auth=AUTH
).json()

found_study = None

for study_id in studies:
    study_info = requests.get(
        f"{ORTHANC_URL}/studies/{study_id}",
        auth=AUTH
    ).json()
    
    patient_id = study_info["PatientMainDicomTags"].get("PatientID")
    
    if patient_id == PATIENT_ID:
        found_study = study_id
        break

if found_study:
    print(" Found study:", found_study)
else:
    print(" Patient not found")

