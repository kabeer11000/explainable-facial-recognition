import cv2
import os
from pathlib import Path
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
from utilities import apply_thresholded_smoothing, estimate_noise_level
# from Model import DecisionTreeClassifier # Assuming this is needed later for training
import joblib
import numpy as np
from skimage.transform import resize
from PIL import Image
from utilities import rotate_pillow_image_from_exif
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time # For timing the execution

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATASET_PATH = Path("C:\\Users\\User\\Documents\\Software\\explainable-facial-recognition\\.ignored\\dataset\\Dataset").resolve()
MAX_IMAGES_PER_PERSON = 50 # Limit to first N images per person for testing/development
FIXED_FACE_SIZE = (128, 128) # Desired (height, width) for resized faces

# --- Helper Function for Image Processing ---
def process_single_image(image_path_str, person_id, haar_detector_paths):
    """
    Processes a single image: reads it, applies smoothing, detects face,
    crops, resizes, and extracts features.
    This function will be run in a separate process.
    """
    image_path = Path(image_path_str) # Convert string back to Path object
    
    # Initialize Haar detectors within each process
    # This avoids pickling issues and ensures each process has its own detector instances.
    v1_process = initialize_haar_detectors(haar_detector_paths)

    try:
        # 1. Read Image
        if person_id in ['10', '11']:
            pil_img = Image.open(str(image_path))
            pil_img = rotate_pillow_image_from_exif(pil_img)
            image = np.array(pil_img)
        else:
            image = cv2.imread(str(image_path))

        if image is None:
            logging.warning(f"Process {os.getpid()}: Could not read image: {image_path.name}")
            return None, None

        # 2. Apply smoothing
        smoothed_image = apply_thresholded_smoothing(image)

        # 3. Detect face using Haar detector
        faces = detect_faces_and_eyes(smoothed_image, v1_process)
        if not faces:
            logging.info(f"Process {os.getpid()}: No face detected in {image_path.name}")
            return None, None

        # 4. Crop and Resize Face
        x, y, w, h = faces[0]['rect']
        cropped_face = smoothed_image[y:y+h, x:x+w]
        cropped_face_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

        cropped_face_gray_resized = resize(cropped_face_gray, FIXED_FACE_SIZE, anti_aliasing=True)
        cropped_face_resized_color = resize(cropped_face, FIXED_FACE_SIZE, anti_aliasing=True)

        # 5. Extract features
        extractor = FeatureExtractor(cropped_face_gray_resized)
        feature_vector = np.concatenate([extractor.calculate().flatten(), cropped_face_resized_color.flatten()])
        
        logging.info(f"Process {os.getpid()}: Successfully processed {image_path.name}")
        return feature_vector, person_id

    except Exception as e:
        logging.error(f"Process {os.getpid()}: Error processing image {image_path.name}: {e}", exc_info=True)
        return None, None

# --- Main Execution ---
def main():
    start_time = time.time()
    
    # Initialize Haar detectors in the main process to get their paths
    # These paths will be passed to child processes for re-initialization
    main_v1_dummy = initialize_haar_detectors() # This call just helps to get the paths
    haar_detector_paths = {key: classifier.getFilename() for key, classifier in main_v1_dummy.items()}
    # Clean up dummy detectors if not needed by main process
    del main_v1_dummy 

    all_features = []
    all_labels = []

    persons = os.listdir(DATASET_PATH)
    persons.sort()

    image_tasks = []
    for person in persons:
        person_dir = DATASET_PATH / person
        if not person_dir.is_dir():
            continue
        logging.info(f"Queueing images for person: {person}")

        images = os.listdir(person_dir)
        images.sort()
        for image_file in images[0:MAX_IMAGES_PER_PERSON]:
            image_path = person_dir / image_file
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            image_tasks.append((str(image_path), person)) # Pass path as string for pickling

    # Use ProcessPoolExecutor for true parallel processing
    num_processes = os.cpu_count() # Optimal for CPU-bound tasks
    logging.info(f"Starting processing with {num_processes} processes...")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all image processing tasks
        # Pass haar_detector_paths to ensure each process can re-initialize its detectors
        future_to_image_info = {
            executor.submit(process_single_image, task_info[0], task_info[1], haar_detector_paths): task_info 
            for task_info in image_tasks
        }

        for future in as_completed(future_to_image_info):
            original_image_path_str, original_person_id = future_to_image_info[future]
            try:
                feature_vector, label = future.result()
                if feature_vector is not None and label is not None:
                    all_features.append(feature_vector)
                    all_labels.append(label)
            except Exception as exc:
                logging.error(f'Image {original_image_path_str} generated an exception during result retrieval: {exc}', exc_info=True)

    # Convert lists to numpy arrays
    if all_features: # Ensure there are features to convert to array
        final_features = np.array(all_features)
        final_labels = np.array(all_labels)

        joblib.dump(final_features, 'features.pkl')
        joblib.dump(final_labels, 'labels.pkl')

        logging.info(f"Features (shape: {final_features.shape}) and labels (shape: {final_labels.shape}) dumped successfully using joblib.")
    else:
        logging.warning("No features were extracted. 'features.pkl' and 'labels.pkl' will not be created.")

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # This block ensures that the main function is only called when the script is executed directly,
    # which is important for multiprocessing on Windows.
    main()