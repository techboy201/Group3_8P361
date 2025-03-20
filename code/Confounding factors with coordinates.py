import os
import cv2
import numpy as np
import random
import pandas as pd

# Folder paths
input_folder = r"C:\Users\20212208\OneDrive - TU Eindhoven\Desktop\8P361\Project_8P\train+val\valid\1"
output_folder = "1_val_modified"
os.makedirs(output_folder, exist_ok=True)

# Create a CSV-file to save dot coordinates
output_csv = os.path.join(output_folder, "stip_coordinates.csv")
stip_data = []

# Get all JPEG images in the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg"))]

for image_file in image_files:
    # Load image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_file}, could not read image.")
        continue

    # Get image dimensions
    height, width, _ = image.shape

    # Random dot size (between 2 and 5 pixels)
    dot_radius = random.randint(2, 5)

    # Random position
    x = random.randint(dot_radius, width - dot_radius)
    y = random.randint(dot_radius, height - dot_radius)

    # Draw black dot
    cv2.circle(image, (x, y), dot_radius, (0, 0, 0), -1)

    # Save modified image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Append data (image filename, x, y, radius)
    stip_data.append([image_file, x, y, dot_radius])

# Save as CSV
df = pd.DataFrame(stip_data, columns=["image_name", "x", "y", "radius"])
df.to_csv(output_csv, index=False)

print(f"Processing complete. Stip coordinates saved in {output_csv}")
