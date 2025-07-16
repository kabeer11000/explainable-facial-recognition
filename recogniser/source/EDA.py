"""
Load images, Rotate other images (10, 11) 

"""
import os
# import matplotlib.pyplot as plt
# from PIL import Image
# import pandas as pd
# import numpy as np
import cv2
from pathlib import Path # Import Path for robust path handling
from tqdm import tqdm

from recogniser.source.feature_extractor import FeatureExtractor
from utilities import apply_thresholded_smoothing # For progress bars, useful for large datasets

# --- 1. Define Dataset Path and Gather Basic Information ---
# The main directory containing subdirectories for each person.
data_dir_path = Path('/home/kabeer/software/explainable-facial-recognition/.ignored/dataset/Dataset').resolve()


# Ensure the data directory exists
if not data_dir_path.exists():
    print(f"Error: Dataset directory not found at {data_dir_path}.")
    print("Please ensure your project structure is:")
    print("  your_project_name/")
    print("  ├── .ignored/")
    print("  │   └── dataset/")
    print("  │       └── Dataset/  <-- Your images are here")
    print("  ├── recogniser/")
    print("  │   └── source/     <-- Your script is here")
    print("  └── ...")
    print("Or update the 'data_dir_path' in the script.")
    exit()

# Each subdirectory in data_dir represents a class (a person).
# We sort the list of IDs for consistent ordering.
# Use iterdir() and filter for directories for robustness
person_folders = sorted([d for d in data_dir_path.iterdir() if d.is_dir()])
person_ids = [p.name for p in person_folders] # Get just the folder names
num_classes = len(person_ids)

print(f"Found {num_classes} persons (classes) in the dataset.")

# --- 2. Loop Over Every Person and Their Images ---
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
total_images_processed = 0

# Loop over every person folder
for person_id in tqdm(person_ids, desc="Processing Persons"):
    person_folder_path = data_dir_path / person_id
    
    # Get all image files within the current person's folder
    # Here, assuming directly in person's folder for simplicity.
    image_files = [
        f for f in person_folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    num_images_for_person = len(image_files)
    print(f"  Person '{person_id}': Found {num_images_for_person} images.")
    
    # Loop over every image for the current person
    for image_path in tqdm(image_files, desc=f"    Images for {person_id}", leave=False):
        # Extract Features here
        image = cv2.imread(image_path)
        
        # Conditionally smoothens image when needed, Requires a lot of RAM
        smoothened_image = apply_thresholded_smoothing(image)

        # Compute Features for each image, and save to file
        fe = FeatureExtractor(smoothened_image)
        partial_features = fe.calculate()
        os.write()
        total_images_processed += 1

print(f"\nFinished iterating through the dataset. Total images encountered: {total_images_processed}")
print(f"Dataset root: {data_dir_path}")

