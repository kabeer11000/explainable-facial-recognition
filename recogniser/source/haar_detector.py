# Define Haar Cascade Detector functions
import numpy as np
import cv2
import os # Import os for path manipulation

HAAR_CASCADES_DIR = os.path.join(os.path.dirname(__file__), '..', 'haar-cascades')

# Global variables to store cascade classifiers once loaded
face_cascade = None
eye_cascade = None

def initialize_haar_detectors():
    """
    Initializes and loads the Haar Cascade classifiers.
    Returns a dictionary of loaded classifiers or None if loading fails.
    """
    global face_cascade, eye_cascade # Declare global to assign to them

    if face_cascade is not None and eye_cascade is not None:
        print("Haar cascade classifiers already initialized.")
        return {"face": face_cascade, "eye": eye_cascade}

    face_cascade_path = os.path.join(HAAR_CASCADES_DIR, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(HAAR_CASCADES_DIR, 'haarcascade_eye.xml')

    print(f"Attempting to load face cascade from: {face_cascade_path}")
    print(f"Attempting to load eye cascade from: {eye_cascade_path}")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier from {face_cascade_path}")
        face_cascade = None # Reset to None if failed
    if eye_cascade.empty():
        print(f"Error: Could not load eye cascade classifier from {eye_cascade_path}")
        eye_cascade = None # Reset to None if failed

    if face_cascade is None or eye_cascade is None:
        return None # Indicate failure
    else:
        print("Successfully loaded Haar cascade classifiers.")
        return {"face": face_cascade, "eye": eye_cascade}


def detect_faces_and_eyes(image_np: np.ndarray, classifier_dict: dict):
    """
    Detects faces and eyes in an image using loaded Haar Cascade classifiers.
    Does not display images.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (BGR).
        classifier_dict (dict): A dictionary containing loaded 'face' and 'eye' classifiers.

    Returns:
        list: A list of detected faces, each as a dictionary
              {'rect': (x, y, w, h), 'eyes': [(ex, ey, ew, eh), ...]}
        None: If classifiers are not provided or image is invalid.
    """
    if image_np is None:
        print("Error: Input image is None.")
        return []

    if classifier_dict is None or "face" not in classifier_dict or "eye" not in classifier_dict:
        print("Error: Haar cascade classifiers not loaded or provided.")
        return []

    face_detector = classifier_dict["face"]
    eye_detector = classifier_dict["eye"]

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5) # Scale factor, minNeighbors

    detected_faces_data = []

    for (x, y, w, h) in faces:
        # Note: In a server context, you typically wouldn't draw directly on the image
        # if you're just returning data. But for clarity, we can prepare a copy
        # for drawing if needed elsewhere.

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_np[y:y+h, x:x+w] # Use the original color ROI for eye detection

        eyes_in_face = []
        # Detect eyes within the detected face region
        eyes = eye_detector.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eyes_in_face.append((ex, ey, ew, eh))

        detected_faces_data.append({
            'rect': (x, y, w, h),
            'eyes': eyes_in_face
        })

    return detected_faces_data

