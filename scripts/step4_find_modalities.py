import requests

ORTHANC = "http://localhost:8042"

STUDIES = [
    "553d7cf2-778939bc-1c16939a-7dab6998-059c46d4",
    "b713ba76-77bbf2d2-562748bd-191fced7-52e159b9",
    "b8d03171-52ef7286-a17c772d-93813c40-be2b5baf"
]

modalities_found = {
    "FLAIR": None,
    "T1": None,
    "T2": None
}

for study_id in STUDIES:
    study_info = requests.get(f"{ORTHANC}/studies/{study_id}").json()
    series_list = study_info.get("Series", [])

    for series_id in series_list:
        series_info = requests.get(f"{ORTHANC}/series/{series_id}").json()
        desc = series_info.get("MainDicomTags", {}).get("SeriesDescription", "").strip().upper()

        print(f"Study {study_id} → Found series: {desc}")

        if "FLAIR" in desc:
            modalities_found["FLAIR"] = series_id
        elif desc == "T1":
            modalities_found["T1"] = series_id
        elif desc == "T2":
            modalities_found["T2"] = series_id

print("\n FINAL MAPPING:")
print(modalities_found)
