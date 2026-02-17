import os
import requests
import zipfile
import shutil
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor

# CONFIG
ORTHANC = "http://localhost:8042"

# You would paste the result from Step 4 here
SERIES = {
    "flair": "42311af6-0627e495-7aae6f38-817661bb-2d4f9871",
    "t1": "39101ca2-9f3a69d9-e5182dc5-5648f5cb-45d07f3a",
    "t2": "f6fc1a44-541c3d2b-1db48635-c5eec6f2-eba3d4b5"
}

MODEL_PATH = r"C:\Users\anvit\Downloads\Suvarna_Model_Run\unet_brats_multimodal_epoch_50.pth"

os.makedirs("dicom_temp", exist_ok=True)
os.makedirs("nifti_temp", exist_ok=True)

# DOWNLOAD & EXTRACT SERIES
def download_series(series_id, name):
    print(f"⬇ Downloading {name}...")
    r = requests.get(f"{ORTHANC}/series/{series_id}/archive")
    zip_path = f"dicom_temp/{name}.zip"
    with open(zip_path, "wb") as f: f.write(r.content)
    extract_path = f"dicom_temp/{name}"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extract_path)
    return extract_path

flair_path = download_series(SERIES["flair"], "flair")
t1_path = download_series(SERIES["t1"], "t1")
t2_path = download_series(SERIES["t2"], "t2")

# CONVERT TO NIFTI
def dicom_to_nifti(dicom_root_folder, output_name):
    import SimpleITK as sitk
    dicom_folder = None
    for root, dirs, files in os.walk(dicom_root_folder):
        if any(file.lower().endswith(".dcm") for file in files):
            dicom_folder = root; break
    
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_folder))
    image = reader.Execute()
    output_path = f"nifti_temp/{output_name}.nii.gz"
    sitk.WriteImage(image, output_path)
    print(f" Saved {output_path}")
    return output_path

flair_nii = dicom_to_nifti(flair_path, "flair")
t1_nii = dicom_to_nifti(t1_path, "t1")
t2_nii = dicom_to_nifti(t2_path, "t2")

# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(spatial_dims=3, in_channels=3, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(" Model loaded")

# LOAD NIFTI + STACK
# FIX: Load the FLAIR object to keep coordinate metadata
flair_obj = nib.load(flair_nii) 
flair_data = flair_obj.get_fdata()
t1_data = nib.load(t1_nii).get_fdata()
t2_data = nib.load(t2_nii).get_fdata()

image = np.stack([flair_data, t1_data, t2_data], axis=0)

transforms = Compose([EnsureChannelFirst(channel_dim=0), ScaleIntensity(), Resize((128, 128, 64)), ToTensor()])
input_tensor = transforms(image).unsqueeze(0).to(device)

# RUN INFERENCE
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

print(" Prediction shape:", prediction.shape)

# SAVE RESULT (FIXED: Copying affine and header from FLAIR)
prediction_nii = nib.Nifti1Image(prediction.astype(np.uint8), flair_obj.affine, flair_obj.header)
nib.save(prediction_nii, "correct_prediction_mask.nii.gz")

print(" Full pipeline complete.\nSaved: correct_prediction_mask.nii.gz")