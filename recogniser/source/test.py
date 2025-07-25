
import cv2
import joblib
import os
from pathlib import Path
from PIL import Image, ExifTags
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Import the wrapper class you defined
from Model import DecisionTreeClassifier 

from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
from utilities import apply_thresholded_smoothing, estimate_noise_level
DATASET_PATH = Path("C:\\Users\\Hp\\Downloads\\uni stuff\\explainable-facial-recognition\\.ignored\\dataset\\Dataset").resolve()

v1= initialize_haar_detectors() 

image= cv2.imread("C:\\Users\\Hp\\Downloads\\uni stuff\\explainable-facial-recognition\\.ignored\\dataset\\Dataset\\01\\frame_00001.jpg")
     

output = detect_faces_and_eyes(image, v1)

# Loop through the detected faces and draw rectangles
for face_data in output:
    (x, y, w, h) = face_data['rect']
    # Draw a rectangle around the face in blue
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Loop through the detected eyes within each face
    for (ex, ey, ew, eh) in face_data['eyes']:
        # The eye coordinates are relative to the face region,
        # so we need to add the face's x and y to get the absolute position.
        cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
showdata= estimate_noise_level(image)
cv2.imshow('Cropped Face', image)
print(f"Estimated noise level: {showdata:.2f}")
cv2.waitKey(0)

cv2.destroyAllWindows()


