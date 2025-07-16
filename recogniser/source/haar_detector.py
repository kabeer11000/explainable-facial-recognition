import numpy as np
import cv2
import os # Import os for path manipulation

# Get the directory of the current script (haar-detector.py)
# This makes paths robust no matter where main.py is run from
SCRIPT_DIR = os.path.dirname(__file__) 
HAAR_CASCADES_DIR = os.path.join(SCRIPT_DIR, '..', 'haar-cascades') # Go up one level, then into haar-cascades

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

# Example usage if you run this script directly (for testing the module)
if __name__ == "__main__":
    classifiers = initialize_haar_detectors()
    if classifiers:
        # Create a dummy image or specify a path to an actual image for testing
        # For a quick test, create a blank image or use a local path
        dummy_image_path = "test_image.jpg"
        # Create a simple dummy image if it doesn't exist
        if not os.path.exists(dummy_image_path):
            dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Add a face here for detection test!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(dummy_image_path, dummy_img)
            print(f"Created a dummy image: {dummy_image_path}. Please replace with an actual face image to test detection.")
        
        img_to_process = cv2.imread(dummy_image_path)
        
        if img_to_process is not None:
            print(f"\nProcessing {dummy_image_path} for face and eye detection...")
            faces_and_eyes = detect_faces_and_eyes(img_to_process, classifiers)
            
            if faces_and_eyes:
                print(f"Found {len(faces_and_eyes)} face(s).")
                # Draw detections on a copy for display
                display_img = img_to_process.copy()
                for face_data in faces_and_eyes:
                    x, y, w, h = face_data['rect']
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue for face
                    for ex, ey, ew, eh in face_data['eyes']:
                        # Adjust eye coordinates relative to the original image
                        cv2.rectangle(display_img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2) # Green for eyes
                
                cv2.imshow('Detected Faces and Eyes', display_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No faces detected in the test image.")
        else:
            print(f"Error: Could not read image from {dummy_image_path}")
