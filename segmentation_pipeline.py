import os
import requests
import zipfile
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor

# CONFIG
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

MODEL_PATH = "model/unet_brats_multimodal_epoch_50.pth"
PATIENT_ID = "001" 

# STEP 1: check_orthanc_connec 
def step1_check_orthanc_connec():
    r = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH)
    print("Status:", r.status_code)
    
    if r.status_code == 200:
        print("Studies:", r.json())
    return r.status_code

# STEP 2,3 : find_patient and get_patient_info 
def step2_find_patient():
    studies = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH).json()
    patient_studies = []
    
    for study_id in studies:
        study_info = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=AUTH).json()
        p_id = study_info["PatientMainDicomTags"].get("PatientID")
        
        if p_id == PATIENT_ID:
            patient_studies.append(study_id)
            
    if patient_studies:
        print(" Found study:", patient_studies[0]) 
        return patient_studies
    else:
        print(" Patient not found")
        return []

# STEP 4: find_modalities 
def step4_find_modalities(study_list):
    modalities_found = {"FLAIR": None, "T1": None, "T2": None}
    
    for study_id in study_list:
        study_info = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=AUTH).json()
        series_list = study_info.get("Series", [])

        for series_id in series_list:
            series_info = requests.get(f"{ORTHANC_URL}/series/{series_id}", auth=AUTH).json()
            desc = series_info.get("MainDicomTags", {}).get("SeriesDescription", "").strip().upper()

            print(f"Study {study_id} → Found series: {desc}")

            if "FLAIR" in desc and not modalities_found["FLAIR"]:
                modalities_found["FLAIR"] = series_id
            elif "T1" in desc and "T1CE" not in desc and not modalities_found["T1"]:
                modalities_found["T1"] = series_id
            elif "T2" in desc and not modalities_found["T2"]:
                modalities_found["T2"] = series_id

    print("\n FINAL MAPPING:")
    print(modalities_found)
    return modalities_found

# STEP 5: run_model 
def step5_run_model(modalities):
    os.makedirs("dicom_temp", exist_ok=True)
    os.makedirs("nifti_temp", exist_ok=True)

    def download_series(series_id, name):
        print(f"⬇ Downloading {name}...")
        r = requests.get(f"{ORTHANC_URL}/series/{series_id}/archive", auth=AUTH)
        zip_path = f"dicom_temp/{name}.zip"
        with open(zip_path, "wb") as f: f.write(r.content)
        extract_path = f"dicom_temp/{name}"
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_path)
        return extract_path

    def dicom_to_nifti(dicom_root, name):
        d_folder = None
        for root, _, files in os.walk(dicom_root):
            if any(f.lower().endswith(".dcm") for f in files):
                d_folder = root; break
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(d_folder))
        sitk.WriteImage(reader.Execute(), f"nifti_temp/{name}.nii.gz")
        print(f" Saved nifti_temp/{name}.nii.gz")
        return f"nifti_temp/{name}.nii.gz"

    # 1. Execution of Data Downloads & Conversion
    f_nii_path = dicom_to_nifti(download_series(modalities["FLAIR"], "flair"), "flair")
    t1_nii_path = dicom_to_nifti(download_series(modalities["T1"], "t1"), "t1")
    t2_nii_path = dicom_to_nifti(download_series(modalities["T2"], "t2"), "t2")

    # 2. AI Inference Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3, 
        in_channels=3, 
        out_channels=2, 
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2), 
        num_res_units=2
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(" Model loaded")

    # 3. Load Data & Prepare for Model
    flair_obj = nib.load(f_nii_path)
    flair_data = flair_obj.get_fdata()
    t1_data = nib.load(t1_nii_path).get_fdata()
    t2_data = nib.load(t2_nii_path).get_fdata()

    img_stack = np.stack([flair_data, t1_data, t2_data], axis=0)
    tx = Compose([EnsureChannelFirst(channel_dim=0), ScaleIntensity(), Resize((128, 128, 64)), ToTensor()])
    input_tensor = tx(img_stack).unsqueeze(0).to(device)

    # 4. Run Prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_low_res = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Upscaling
    print(f"Resizing mask from {pred_low_res.shape} to clinical resolution {flair_data.shape}...")
    
    zoom_factors = (
        flair_data.shape[0] / pred_low_res.shape[0],
        flair_data.shape[1] / pred_low_res.shape[1],
        flair_data.shape[2] / pred_low_res.shape[2]
    )
    
    pred_high_res = zoom(pred_low_res, zoom_factors, order=0)

    # 5. Saving with High-Res Data + Original Metadata
    prediction_nii = nib.Nifti1Image(
        pred_high_res.astype(np.uint8), 
        flair_obj.affine, 
        flair_obj.header
    )
    
    output_filename = "prediction_mask.nii.gz"
    nib.save(prediction_nii, output_filename)
    print(f" Full pipeline complete.\nSaved: {output_filename} with coordinate and resolution alignment.")


# MAIN CONTROL 
if __name__ == "__main__":
    if step1_check_orthanc_connec() == 200:
        studies = step2_find_patient()
        if studies:
            mapping = step4_find_modalities(studies)
            if all(mapping.values()):
                step5_run_model(mapping)
            else:

                print(" Still missing some modalities in the mapping.")



