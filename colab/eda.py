import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

# =============================================================================
# Task 1: Data Loading and Preparation (Assigned to Kabeer)
# =============================================================================

def load_and_prepare_data(data_dir):
    """
    Loads data from the specified directory, identifies class labels,
    and counts the number of images per class.

    Args:
        data_dir (str): The path to the main dataset directory.

    Returns:
        tuple: A tuple containing:
            - person_ids (list): A sorted list of the class labels (person IDs).
            - image_counts (dict): A dictionary mapping each person ID to their image count.
    """
    print("--- Task by Kabeer: Loading and Preparing Data ---")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at {data_dir}")
        return None, None

    # Each subdirectory in data_dir represents a class (a person).
    person_ids = sorted(os.listdir(data_dir))
    num_classes = len(person_ids)

    print(f"Found {num_classes} classes (individuals).")
    print(f"Class labels (Person IDs): {person_ids}")

    # Create a dictionary to store the number of images for each person.
    image_counts = {}
    total_images = 0
    
    print("\nCounting images for each person...")
    for person_id in person_ids:
        person_folder = os.path.join(data_dir, person_id)
        if os.path.isdir(person_folder):
            count = len(os.listdir(person_folder))
            image_counts[person_id] = count
            total_images += count
    
    print(f"Total number of images in the dataset: {total_images}\n")
    return person_ids, image_counts

# =============================================================================
# Task 2: Data Visualization (Assigned to Reham)
# =============================================================================

def visualize_class_distribution(image_counts):
    """
    Creates and displays a bar chart showing the number of images per person.

    Args:
        image_counts (dict): A dictionary mapping person IDs to image counts.
    """
    print("--- Task by Reham: Visualizing Class Distribution ---")
    plt.figure(figsize=(12, 6))
    plt.bar(image_counts.keys(), image_counts.values(), color='mediumpurple')
    plt.title('Distribution of Images per Person', fontsize=16)
    plt.xlabel('Person ID', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def visualize_sample_images(data_dir, person_ids):
    """
    Displays a grid of sample images, one from each class.

    Args:
        data_dir (str): The path to the main dataset directory.
        person_ids (list): A list of the class labels (person IDs).
    """
    print("\n--- Task by Reham: Displaying Sample Images ---")
    # Create a 3x4 grid, accommodating up to 12 classes.
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Sample Image from Each Class', fontsize=20)
    axes = axes.ravel() # Flatten for easy iteration

    for i, person_id in enumerate(person_ids):
        sample_image_path = os.path.join(data_dir, person_id, os.listdir(os.path.join(data_dir, person_id))[0])
        img = Image.open(sample_image_path)
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Person: {person_id}', fontsize=14)
        ax.axis('off')

    # Hide unused subplots
    for j in range(len(person_ids), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =============================================================================
# Task 3: In-depth Analysis and Reporting (Assigned to Hiba)
# =============================================================================

def analyze_image_properties(data_dir, person_ids):
    """
    Analyzes a sample image from each class to check for consistent
    properties like dimensions and color mode.

    Args:
        data_dir (str): The path to the main dataset directory.
        person_ids (list): A list of the class labels.
    """
    print("\n--- Task by Hiba: Analyzing Image Properties ---")
    image_shapes = set()
    image_modes = set()

    for person_id in person_ids:
        first_image_file = os.listdir(os.path.join(data_dir, person_id))[0]
        image_path = os.path.join(data_dir, person_id, first_image_file)
        with Image.open(image_path) as img:
            image_shapes.add(img.size)  # (width, height)
            image_modes.add(img.mode)   # e.g., 'RGB'

    print(f"Unique image dimensions (width, height) found: {image_shapes}")
    print(f"Unique image modes (color channels) found: {image_modes}")

    if len(image_shapes) == 1 and len(image_modes) == 1:
        print("Conclusion: All sample images have consistent properties.")
    else:
        print("Warning: Inconsistent image properties found. Preprocessing will be required.")

def generate_summary_report(image_counts):
    """
    Creates and prints a formatted summary table of the EDA findings.

    Args:
        image_counts (dict): A dictionary mapping person IDs to image counts.
    """
    print("\n--- Task by Hiba: Generating Final EDA Summary Report ---")
    summary_df = pd.DataFrame(list(image_counts.items()), columns=['Person ID', 'Image Count'])
    print(summary_df.to_string(index=False))

# =============================================================================
# Main Execution Block: Orchestrating the EDA Process
# =============================================================================

if __name__ == "__main__":
    # Define the path to the dataset
    DATA_DIRECTORY = '/content/dataset/Dataset/'

    # --- Step 1: Kabeer's contribution ---
    # Load the data and get the foundational data structures
    person_ids, image_counts = load_and_prepare_data(DATA_DIRECTORY)

    # Proceed only if data was loaded successfully
    if person_ids and image_counts:
        
        # --- Step 2: Hiba's contribution (Analysis) ---
        # Analyze technical properties of the images
        analyze_image_properties(DATA_DIRECTORY, person_ids)

        # --- Step 3: Reham's contribution ---
        # Create visualizations based on the data
        visualize_class_distribution(image_counts)
        visualize_sample_images(DATA_DIRECTORY, person_ids)

        # --- Step 4: Hiba's contribution (Reporting) ---
        # Present the final summary table
        generate_summary_report(image_counts)
        
        print("\nExploratory Data Analysis complete.")

