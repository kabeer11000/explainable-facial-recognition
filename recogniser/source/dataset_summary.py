# Expects Google Colab Notebook
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

# --- 1. Define Dataset Path and Gather Basic Information ---
# The main directory containing subdirectories for each person.
data_dir = "C:\Users\Hp\Downloads\uni stuff\Dataset"

# Each subdirectory in data_dir represents a class (a person).
# We sort the list of IDs for consistent ordering.
person_ids = sorted(os.listdir(data_dir))
num_classes = len(person_ids)

print("--- Basic Dataset Information ---")
print(f"Found {num_classes} classes (individuals).")
print(f"Class labels (Person IDs): {person_ids}")

# --- 2. Count Images Per Class and Total Images ---
# We'll create a dictionary to store the number of images for each person.
# This is important for identifying any class imbalance.
image_counts = {}
total_images = 0

print("\n--- Counting Images ---")
for person_id in person_ids:
    person_folder = os.path.join(data_dir, person_id)
    # We list the files in each person's directory to get the count.
    if os.path.isdir(person_folder):
        count = len(os.listdir(person_folder))
        image_counts[person_id] = count
        total_images += count

print(f"Total number of images in the dataset: {total_images}")
print("\nNumber of images per person:")
for person_id, count in image_counts.items():
    print(f"  - Person '{person_id}': {count} images")

# --- 3. Visualize Class Distribution ---
# A bar chart provides a clear visual representation of the image counts per class.
print("\n--- Visualizing Class Distribution ---")
plt.figure(figsize=(12, 6))
plt.bar(image_counts.keys(), image_counts.values(), color='teal')
plt.title('Distribution of Images per Person', fontsize=16)
plt.xlabel('Person ID', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- 4. Analyze Image Properties (Size and Color Mode) ---
# We check a sample image from each class to ensure consistency.
# If images have different sizes or color modes, they will need to be standardized.
print("\n--- Analyzing Image Properties ---")
image_shapes = set()
image_modes = set()

# Check the first image from each class
for person_id in person_ids:
    person_folder = os.path.join(data_dir, person_id)
    first_image_file = os.listdir(person_folder)[0]
    image_path = os.path.join(person_folder, first_image_file)

    with Image.open(image_path) as img:
        image_shapes.add(img.size)  # (width, height)
        image_modes.add(img.mode)   # e.g., 'RGB', 'L' (grayscale)

print(f"Unique image dimensions (width, height) found: {image_shapes}")
print(f"Unique image modes (color channels) found: {image_modes}")

if len(image_shapes) == 1 and len(image_modes) == 1:
    print("\nConclusion: All sample images have consistent dimensions and color modes.")
    sample_shape = list(image_shapes)[0]
    sample_mode = list(image_modes)[0]
    print(f"  - Image Size: {sample_shape[0]}x{sample_shape[1]} pixels")
    print(f"  - Image Mode: {sample_mode}")
else:
    print("\nWarning: Inconsistent image properties found. Preprocessing (resizing and color conversion) will be required.")

# --- 5. Visualize Sample Images from Each Class ---
# Displaying samples helps in understanding the visual characteristics of the data,
# such as lighting, pose, and background.
print("\n--- Displaying Sample Images from Each Class ---")

# Create a grid of subplots to display one image per class.
# We'll use a 3x4 grid, which can accommodate up to 12 classes.
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Sample Image from Each Class', fontsize=20)
axes = axes.ravel() # Flatten the 2D array of axes for easy iteration

for i, person_id in enumerate(person_ids):
    # Get the path to the first image for the current person
    sample_image_path = os.path.join(data_dir, person_id, os.listdir(os.path.join(data_dir, person_id))[0])
    img = Image.open(sample_image_path)

    # Display the image on its subplot
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f'Person: {person_id}', fontsize=14)
    ax.axis('off') # Hide the x and y axes

# Hide any unused subplots if num_classes is less than 12
for j in range(num_classes, len(axes)):
    axes[j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for the suptitle
plt.show()

# --- 6. Final EDA Summary Table ---
# A pandas DataFrame is an excellent tool for presenting a clean summary.
eda_summary_df = pd.DataFrame({
    'Person ID': image_counts.keys(),
    'Image Count': image_counts.values()
})

print("\n--- Exploratory Data Analysis (EDA) Summary Table ---")
print(eda_summary_df.to_string(index=False))