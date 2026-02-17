import requests

ORTHANC = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")
PATIENT_ID = "001"

# Automatically find all studies for this patient
all_studies = requests.get(f"{ORTHANC}/studies", auth=AUTH).json()
STUDIES = []

for s_id in all_studies:
    s_info = requests.get(f"{ORTHANC}/studies/{s_id}", auth=AUTH).json()
    if s_info["PatientMainDicomTags"].get("PatientID") == PATIENT_ID:
        STUDIES.append(s_id)

modalities_found = {"FLAIR": None, "T1": None, "T2": None}

for study_id in STUDIES:
    study_info = requests.get(f"{ORTHANC}/studies/{study_id}", auth=AUTH).json()
    series_list = study_info.get("Series", [])

    for series_id in series_list:
        series_info = requests.get(f"{ORTHANC}/series/{series_id}", auth=AUTH).json()
        desc = series_info.get("MainDicomTags", {}).get("SeriesDescription", "").strip().upper()
        print(f"Study {study_id} → Found series: {desc}")

        if "FLAIR" in desc and not modalities_found["FLAIR"]:
            modalities_found["FLAIR"] = series_id
        elif desc == "T1" and not modalities_found["T1"]:
            modalities_found["T1"] = series_id
        elif desc == "T2" and not modalities_found["T2"]:
            modalities_found["T2"] = series_id

print("\n FINAL MAPPING:")
print(modalities_found)