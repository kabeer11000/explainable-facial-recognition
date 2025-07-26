import cv2
import joblib
import numpy as np
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
from utilities import apply_thresholded_smoothing, estimate_noise_level, rotate_pillow_image_from_exif
from skimage.transform import resize
from PIL import Image

# Load the trained model
clf = joblib.load("decision_tree_model.pkl")
from sklearn.preprocessing import LabelEncoder
le = joblib.load("label_encoder.pkl")

# Path to the image you want to predict
image_path = "C:\\Users\\Hp\\Downloads\\uni stuff\\explainable-facial-recognition\\recogniser\\frame_00284.jpg"

# If the image is from person 10 or 11 and needs rotation, use PIL and your rotation function
# Otherwise, use OpenCV directly
# For demo, let's use OpenCV:
image = cv2.imread(image_path)
if image is None:
    print("Could not read image.")
    exit()

# Optionally, apply rotation if needed:
# pil_img = Image.open(image_path)
# pil_img = rotate_pillow_image_from_exif(pil_img)
# image = np.array(pil_img)

# 1. Estimate noise level (optional, for info)
noise_level = estimate_noise_level(image)
print(f"Noise level for input image: {noise_level}")

# 2. Apply smoothing
smoothed_image = apply_thresholded_smoothing(image)

# 3. Detect face
v1 = initialize_haar_detectors()
faces = detect_faces_and_eyes(smoothed_image, v1)
if not faces:
    print("No face detected in the image.")
    exit()

# 4. Crop face (use first detected face)
x, y, w, h = faces[0]['rect']
cropped_face = smoothed_image[y:y+h, x:x+w]

# 5. Convert cropped face to grayscale
cropped_face_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

# 6. Resize to match training
fixed_size = (128, 128)
cropped_face_gray = resize(cropped_face_gray, fixed_size, anti_aliasing=True)

# 7. Extract features
extractor = FeatureExtractor(cropped_face_gray)
feature_vector = extractor.calculate()

# 8. Predict
feature_vector = np.array([feature_vector])  # Convert to NumPy array with shape (1, n_features)
#predicted_person = clf.predict(feature_vector)[0]
#print("Predicted person:", predicted_person)
predicted_label = clf.predict(feature_vector)[0]
predicted_person = le.inverse_transform([predicted_label])[0]  # to Get the actual label like '10' or '11'
print("Predicted person:", predicted_person)
# Display the cropped face
cv2.imshow('Cropped Face', cropped_face_gray)
cv2.waitKey(0)
