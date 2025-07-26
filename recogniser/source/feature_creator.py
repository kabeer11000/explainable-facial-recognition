import cv2
import os
from pathlib import Path
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
from utilities import apply_thresholded_smoothing, estimate_noise_level
from Model import DecisionTreeClassifier
import joblib
import numpy as np
from skimage.transform import resize
from PIL import Image
from utilities import rotate_pillow_image_from_exif


# Load the trained model
#clf = joblib.load("decision_tree_model.pkl")

DATASET_PATH = Path("C:\\Users\\User\\Documents\\Software\\explainable-facial-recognition\\.ignored\\dataset\\Dataset").resolve()

# Initialize Haar detectors
v1 = initialize_haar_detectors()

features = []
labels = []

persons = os.listdir(DATASET_PATH)
persons.sort()

for person in persons:
    person_dir = DATASET_PATH / person
    if not person_dir.is_dir():
        continue
        print(f"Processing person: {person}")

    images = os.listdir(person_dir)[0:50]
    images.sort()
    for image_file in images:  # Limit to first 10 images for testing
        image_path = person_dir / image_file
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            print(f"Reading image: {image_file}")
       # image = cv2.imread(str(image_path))
       # if image is None:
        #    print(f"Could not read image: {image_file}")
         #   continue
        
   # For person 10 and 11, apply EXIF-based rotation using PIL
        if person in ['10', '11']:
            pil_img = Image.open(str(image_path))
            pil_img = rotate_pillow_image_from_exif(pil_img)
            image = np.array(pil_img)  # Convert back to NumPy array for OpenCV
        else:
            image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_file}")
            continue

        

        # 1. Estimate noise level
        noise_level = estimate_noise_level(image)
        print(f"Noise level for {image_file}: {noise_level}")

        # 2. Apply smoothing
        smoothed_image = apply_thresholded_smoothing(image)
        print(f"Smoothing applied for {image_file}")

        # 3. Detect face using Haar detector
        faces = detect_faces_and_eyes(smoothed_image, v1)
        if not faces:
            continue  # Skip if no face detected

        # 4. Crop face (use first detected face)
        x, y, w, h = faces[0]['rect']
        cropped_face = smoothed_image[y:y+h, x:x+w]
        print(f"Cropped face for {image_file}")
        
        
        cropped_face_gray = cropped_face


        # Convert cropped face to grayscale
        cropped_face_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        print(f"Converted cropped face to grayscale for {image_file}")

        # Kabeer's Addition
        # Resize the cropped face (grayscale) to a fixed size — recommended size for faces
        fixed_size = (128, 128)  # (height, width) — balanced resolution

        # Use anti_aliasing to smooth during resize
        cropped_face_gray = resize(cropped_face_gray, fixed_size, anti_aliasing=True)
        cropped_face_resized = resize(cropped_face, fixed_size, anti_aliasing=True)
        # cv2.imwrite('./cropped-test-image.jpg', (cropped_face_gray * 255).astype(np.uint8))
        # cv2.imshow('Cropped Face', cropped_face_gray)
        # cv2.waitKey(0)
        # break
        # 5. Extract features
        extractor = FeatureExtractor(cropped_face_gray)
        feature_vector = np.concatenate([extractor.calculate().flatten(), cropped_face_resized.flatten()])
        print(feature_vector.shape)
        print(f"Extracted features for {image_file}")
        features.append(feature_vector)
        labels.append(person)
        

# Before training
features = np.array(features)
labels = np.array(labels)

joblib.dump(features, 'features.pkl')
joblib.dump(labels, 'labels.pkl')

print("Features and labels dumped successfully using joblib.")

