# @deprecated
import cv2
import os
from pathlib import Path
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
from utilities import apply_thresholded_smoothing, estimate_noise_level, rotate_pillow_image_from_exif
from Model import DecisionTreeClassifier
import joblib
import numpy as np
from skimage.transform import resize
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Container for image processing results"""
    feature_vector: np.ndarray
    label: str
    image_path: str
    success: bool
    error_message: str = ""

class FaceProcessingPipeline:
    """Multithreaded facial recognition preprocessing pipeline"""
    
    def __init__(self, dataset_path: str, max_workers: int = 4, images_per_person: int = 50):
        self.dataset_path = Path(dataset_path).resolve()
        self.max_workers = max_workers
        self.images_per_person = images_per_person
        self.fixed_size = (128, 128)
        self.valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        # Thread-safe collections
        self.features = []
        self.labels = []
        self.lock = Lock()
        
        # Initialize Haar detectors (thread-safe after initialization)
        self.haar_detectors = initialize_haar_detectors()
        
    def process_single_image(self, image_path: Path, person_label: str) -> ProcessingResult:
        """Process a single image and extract features"""
        try:
            # Handle EXIF rotation for specific persons
            if person_label in ['10', '11']:
                pil_img = Image.open(str(image_path))
                pil_img = rotate_pillow_image_from_exif(pil_img)
                image = np.array(pil_img)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                return ProcessingResult(
                    feature_vector=None,
                    label=person_label,
                    image_path=str(image_path),
                    success=False,
                    error_message="Could not read image"
                )
            
            # Estimate noise level
            noise_level = estimate_noise_level(image)
            
            # Apply smoothing
            smoothed_image = apply_thresholded_smoothing(image)
            
            # Detect faces
            faces = detect_faces_and_eyes(smoothed_image, self.haar_detectors)
            if not faces:
                return ProcessingResult(
                    feature_vector=None,
                    label=person_label,
                    image_path=str(image_path),
                    success=False,
                    error_message="No face detected"
                )
            
            # Crop face (use first detected face)
            x, y, w, h = faces[0]['rect']
            cropped_face = smoothed_image[y:y+h, x:x+w]
            
            # Convert to grayscale
            cropped_face_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
            
            # Resize both grayscale and color versions
            cropped_face_gray_resized = resize(cropped_face_gray, self.fixed_size, anti_aliasing=True)
            cropped_face_resized = resize(cropped_face, self.fixed_size, anti_aliasing=True)
            
            # Extract features
            extractor = FeatureExtractor(cropped_face_gray_resized)
            feature_vector = np.concatenate([
                extractor.calculate().flatten(), 
                cropped_face_resized.flatten()
            ])
            
            return ProcessingResult(
                feature_vector=feature_vector,
                label=person_label,
                image_path=str(image_path),
                success=True
            )
            
        except Exception as e:
            return ProcessingResult(
                feature_vector=None,
                label=person_label,
                image_path=str(image_path),
                success=False,
                error_message=str(e)
            )
    
    def collect_image_tasks(self) -> List[Tuple[Path, str]]:
        """Collect all image processing tasks"""
        tasks = []
        persons = sorted(os.listdir(self.dataset_path))
        
        for person in persons:
            person_dir = self.dataset_path / person
            if not person_dir.is_dir():
                continue
                
            images = sorted(os.listdir(person_dir))
            for image_file in images[:self.images_per_person]:
                image_path = person_dir / image_file
                if image_path.suffix.lower() in self.valid_extensions:
                    tasks.append((image_path, person))
        
        return tasks
    
    def process_images_parallel(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process all images using multithreading"""
        tasks = self.collect_image_tasks()
        total_tasks = len(tasks)
        
        logger.info(f"Processing {total_tasks} images using {self.max_workers} workers")
        
        successful_results = []
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.process_single_image, image_path, person): (image_path, person)
                for image_path, person in tasks
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_task), 1):
                result = future.result()
                
                if result.success:
                    successful_results.append(result)
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process {result.image_path}: {result.error_message}")
                
                # Progress logging
                if i % 50 == 0 or i == total_tasks:
                    logger.info(f"Processed {i}/{total_tasks} images "
                              f"(Success: {len(successful_results)}, Failed: {failed_count})")
        
        if not successful_results:
            raise ValueError("No images were successfully processed")
        
        # Extract features and labels from successful results
        features = np.array([result.feature_vector for result in successful_results])
        labels = np.array([result.label for result in successful_results])
        
        logger.info(f"Successfully processed {len(successful_results)} images")
        logger.info(f"Feature matrix shape: {features.shape}")
        
        return features, labels
    
    def save_results(self, features: np.ndarray, labels: np.ndarray, 
                    features_path: str = 'features.pkl', labels_path: str = 'labels.pkl'):
        """Save features and labels to disk"""
        joblib.dump(features, features_path)
        joblib.dump(labels, labels_path)
        logger.info(f"Features saved to {features_path}")
        logger.info(f"Labels saved to {labels_path}")

def main():
    """Main execution function"""
    DATASET_PATH = "C:\\Users\\User\\Documents\\Software\\explainable-facial-recognition\\.ignored\\dataset\\Dataset"
    
    # Configure pipeline
    pipeline = FaceProcessingPipeline(
        dataset_path=DATASET_PATH,
        max_workers=12,  # Adjust based on your CPU cores
        images_per_person=50
    )
    
    try:
        # Process images
        features, labels = pipeline.process_images_parallel()
        
        # Save results
        pipeline.save_results(features, labels)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()