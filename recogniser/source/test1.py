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
