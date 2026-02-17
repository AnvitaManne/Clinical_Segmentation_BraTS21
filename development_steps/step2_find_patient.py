import requests

ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

PATIENT_ID = "001" 

studies = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH).json()
found_studies = []

for study_id in studies:
    study_info = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=AUTH).json()
    patient_id = study_info["PatientMainDicomTags"].get("PatientID")
    
    if patient_id == PATIENT_ID:
        found_studies.append(study_id)

if found_studies:
    print("Found studies for Patient 001:", found_studies)
else:
    print("Patient not found")