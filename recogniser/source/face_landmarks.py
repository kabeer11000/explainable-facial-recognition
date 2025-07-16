# Facial Landmarks
import cv2
import dlib
import numpy as np
import imutils
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
DLIB_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat" 
TARGET_FACE_SIZE = (160, 160) # Common size for many face recognition models
FACE_ALIGNMENT_PADDING = 0.2 # Add 20% padding around the aligned face to avoid cropping features


class FacePreprocessor:
    """
    Combines face detection, orientation correction, cropping, alignment, and resizing.
    """
    def __init__(self, predictor_path: str = DLIB_LANDMARK_PREDICTOR, 
                 target_size: tuple[int, int] = TARGET_FACE_SIZE,
                 padding: float = FACE_ALIGNMENT_PADDING):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError as e:
            print(f"Error loading dlib shape predictor: {e}")
            print("Please ensure 'shape_predictor_68_face_landmarks.dat' is downloaded and in the correct path.")
            print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Then extract: bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
            raise SystemExit("Exiting due to missing dlib shape predictor.")

        self.target_size = target_size
        self.padding = padding # Padding for imutils.align_face

    def process_face(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detects, corrects orientation, aligns, crops, and resizes a face in an image.
        Assumes one primary face per image.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            np.ndarray | None: Processed face image (BGR format, target_size) or None if no face.
        """
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        if len(faces) == 0:
            return None # No face found

        # Prioritize the largest face
        main_face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Perform alignment using imutils.align_face for best results
        # This function handles both rotation and translation based on eye landmarks.
        aligned_face = imutils.align_face(image, gray, main_face, self.predictor, 
                                          desiredFaceWidth=self.target_size[0], 
                                          desiredFaceHeight=self.target_size[1],
                                          desiredFacePadding=self.padding)
        
        return aligned_face

    def preprocess_dataset(self, input_dir: Path, output_dir: Path):
        """
        Processes all images in the input directory, aligns faces, and saves them.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        all_image_paths = [p for p in input_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_extensions]

        print(f"Preprocessing {len(all_image_paths)} images from {input_dir} for alignment.")
        for img_path in tqdm(all_image_paths, desc="Aligning Faces"):
            relative_path = img_path.relative_to(input_dir)
            output_img_path = output_dir / relative_path
            output_img_path.parent.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(str(img_path))
            processed_face = self.process_face(img)

            if processed_face is not None:
                cv2.imwrite(str(output_img_path), processed_face)
            else:
                # Optionally, copy original if no face detected or log it
                # print(f"Skipped {img_path}: No face detected.")
                pass # Skip saving if no face detected


class AverageFaceGenerator:
    """
    Generates and displays average faces for each person in a pre-aligned dataset.
    """
    def __init__(self, aligned_dataset_path: Path):
        self.aligned_dataset_path = aligned_dataset_path

    def generate_average_faces(self) -> dict[str, np.ndarray]:
        """
        Generates an average face for each person (subfolder) in the dataset.

        Returns:
            dict[str, np.ndarray]: A dictionary where keys are person IDs and values
                                   are their average face images.
        """
        average_faces = {}
        
        person_folders = [d for d in self.aligned_dataset_path.iterdir() if d.is_dir()]
        
        if not person_folders:
            print(f"No person folders found in {self.aligned_dataset_path}. Please preprocess first.")
            return average_faces

        for person_folder in tqdm(person_folders, desc="Calculating Average Faces"):
            person_id = person_folder.name
            person_images = []
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            for img_path in person_folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Ensure all images are of the same size before averaging
                        if img.shape[:2] == TARGET_FACE_SIZE: # Assuming already preprocessed to TARGET_FACE_SIZE
                            person_images.append(img.astype(np.float32)) # Convert to float for averaging
                        else:
                            print(f"Warning: Image {img_path} has inconsistent size {img.shape[:2]}. Skipping.")

            if len(person_images) > 0:
                # Sum all images and divide by count
                sum_images = np.sum(person_images, axis=0)
                average_face = (sum_images / len(person_images)).astype(np.uint8)
                average_faces[person_id] = average_face
            else:
                print(f"No valid images found for {person_id}. Skipping average face generation.")
                
        return average_faces

    def display_average_faces(self, avg_faces_dict: dict[str, np.ndarray]):
        """
        Displays the generated average faces.
        """
        if not avg_faces_dict:
            print("No average faces to display.")
            return

        # Determine grid size
        num_faces = len(avg_faces_dict)
        cols = int(np.ceil(np.sqrt(num_faces)))
        rows = int(np.ceil(num_faces / cols))

        plt.figure(figsize=(cols * 3, rows * 3)) # Adjust figure size

        sorted_person_ids = sorted(avg_faces_dict.keys())

        for i, person_id in enumerate(sorted_person_ids):
            avg_face = avg_faces_dict[person_id]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(avg_face, cv2.COLOR_BGR2RGB))
            plt.title(f"Avg Face: {person_id}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

