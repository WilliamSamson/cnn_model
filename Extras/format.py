import os
import pandas as pd
import shutil

from pathlib import Path

# Load Model & Scaler
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "Training_Set"/ "plate_numbers.csv"
data_path2 = project_root / "Training_Set"/ "plates" / "Plates"

if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")

if not data_path2.exists():
    raise FileNotFoundError(f"File not found: {data_path2}")


# Paths
csv_path = data_path
images_folder = data_path2 # Destination folder

# Ensure folder exists
os.makedirs(images_folder, exist_ok=True)

# Load and process CSV
df = pd.read_csv(csv_path)
df["image_name"] = df["image_path"].apply(lambda x: os.path.basename(x))  # Extract filename
df.drop(columns=["image_path"], inplace=True)  # Remove path column

# Save cleaned CSV
cleaned_csv_path = "formatted_plates.csv"
df.to_csv(cleaned_csv_path, index=False)
print(f"Formatted dataset saved to {cleaned_csv_path}")

# Move and rename images
for img_name in df["image_name"]:
    original_path = os.path.join("/content/drive/MyDrive/Plates", img_name)  # Change path if needed
    new_path = os.path.join(images_folder, img_name)

    if os.path.exists(original_path):  # Check if file exists before moving
        shutil.move(original_path, new_path)

print(f"All images moved and organized in '{images_folder}'")
