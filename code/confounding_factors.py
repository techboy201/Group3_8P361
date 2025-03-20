import os
import cv2
import numpy as np
import random

# Folder paths
input_folder = r"C:\Users\20212208\OneDrive - TU Eindhoven\Desktop\8P361\Project_8P\train+val\valid\1"
output_folder = "1_val_modified"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

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

    # Random dot size (between 5 and 20 pixels)
    dot_radius = random.randint(1, 5)

    # Random position
    x = random.randint(dot_radius, width - dot_radius)
    y = random.randint(dot_radius, height - dot_radius)

    # Draw black dot
    cv2.circle(image, (x, y), dot_radius, (0, 0, 0), -1)

    # Save modified image with JPEG quality set to 90 (adjustable)
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

print("Processing complete. Modified images saved in '1_modified'.")
