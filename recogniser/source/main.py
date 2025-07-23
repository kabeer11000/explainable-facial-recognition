#  pipeline: Data Loading, Preprocessing, Feature Extraction, Training, Evaluation, and Server Setup

import asyncio
import websockets
from aiohttp import web
import json
import base64
import numpy as np
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
from utilities import apply_thresholded_smoothing

# --- Configuration ---
IMAGE_SIZE = (256, 256)
# Updated dataset path based on user's request (specific to local environment/Git structure)
DATASET_PATH = Path('/home/kabeer/software/explainable-facial-recognition/.ignored/dataset/Dataset').resolve()
MODEL_PATH = 'decision_tree_model.joblib'
ENCODER_PATH = 'label_encoder.joblib'

# Global variables to store trained model, encoder, and classifiers
dt_model = None
label_encoder = None
classifiers = None

# --- Helper Functions (from previous cells/discussion) ---

def correct_image_rotation(image_path):
    """Corrects image rotation based on EXIF data."""
    try:
        img = Image.open(image_path)
        for orientation_tag in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation_tag] == 'Orientation': break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation_tag)
            if orientation == 3: img = img.rotate(180, expand=True)
            elif orientation == 6: img = img.rotate(-90, expand=True)
            elif orientation == 8: img = img.rotate(90, expand=True)
        return np.array(img)
    except Exception:
        # Fallback to cv2.imread if PIL/EXIF fails
        return cv2.imread(str(image_path))

# apply_thresholded_smoothing is imported from utilities
# detect_faces_and_eyes is imported from haar_detector
# FeatureExtractor is imported from feature_extractor

# --- Data Loading, Preprocessing, Feature Extraction, Training ---

def train_face_recognition_model():
    """
    Loads data, preprocesses, extracts features, trains Decision Tree model,
    evaluates, and saves the model and label encoder.
    Implements the training pipeline steps.
    """
    global classifiers, dt_model, label_encoder

    print("Starting model training pipeline...")

    # 1. Initialize Haar cascades (if not already)
    # This uses the path relative to the script (__file__) defined in the haar_detector module.
    if classifiers is None:
        classifiers = initialize_haar_detectors()
        if classifiers is None:
            raise RuntimeError("Failed to load Haar cascades during training.")

    features = []
    labels = []
    image_extensions = {'.png', '.jpg', '.jpeg'}

    # 2. Load and Preprocess Dataset
    print(f"Loading images from {DATASET_PATH}...")
    for person_folder in DATASET_PATH.iterdir():
        if not person_folder.is_dir(): continue
        person_name = person_folder.name

        # Filter for image files with correct extensions
        img_files = [f for f in person_folder.iterdir() if f.suffix.lower() in image_extensions]
        print(f"Found {len(img_files)} images for person: {person_name}")

        for img_path in img_files:
            # a. Correct image rotation (Pipeline Step 1)
            img_np = correct_image_rotation(img_path)
            if img_np is None:
                print(f"Warning: Could not load or correct rotation for {img_path}, skipping.")
                continue

            # Ensure image is BGR for OpenCV operations if it came from PIL (often RGB)
            if len(img_np.shape) == 3 and img_np.shape[-1] == 3 and img_np.dtype == np.uint8:
                 # Simple check - if it's uint8 and 3 channels, assume it might be RGB from PIL
                 # A more robust check would involve checking the color space explicitly if possible
                 try:
                     # Attempt conversion - if it fails, it was likely already BGR
                      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                 except:
                      pass # Image was likely already BGR or grayscale


            # b. Apply smoothing based on noise (Pipeline Step 2)
            img_processed = apply_thresholded_smoothing(img_np)
            if img_processed is None:
                 print(f"Warning: Smoothing failed for {img_path}, skipping.")
                 continue

            # c. Detect faces and eyes (Pipeline Step 3: Face Detection)
            faces = detect_faces_and_eyes(img_processed, classifiers)
            if not faces:
                # print(f"No face found in {img_path}, skipping.") # Too verbose
                continue

            # d. Crop first face and resize (Pipeline Step 3: Image Cropping)
            x, y, w, h = faces[0]['rect']
            cropped_face = img_processed[y:y+h, x:x+w]
            # Ensure cropped_face is not empty
            if cropped_face.size == 0:
                print(f"Warning: Cropped face is empty for {img_path}, skipping.")
                continue

            resized_face = cv2.resize(cropped_face, IMAGE_SIZE)

             # Convert to grayscale for feature extraction if not already
            if len(resized_face.shape) == 3 and resized_face.shape[-1] == 3:
                 resized_face_gray = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            else:
                 resized_face_gray = resized_face # Already grayscale


            # e. Extract features (Pipeline Step 4: Feature Extraction)
            fe = FeatureExtractor(resized_face_gray)
            fv = fe.calculate(visulalize=False) # Set visualize=True for debugging feature extraction
            features.append(fv)
            labels.append(person_name)

    features = np.array(features)
    labels = np.array(labels)

    if len(features) == 0:
        raise ValueError("No faces with extractable features found in the dataset.")

    print(f"Finished preprocessing and feature extraction. Total samples: {len(labels)}")
    print(f"Feature vector shape: {features.shape}")


    # 3. Split Data (80/20 per person) 
    # Use LabelEncoder before splitting for stratify
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Stratified split to ensure 80/20 per class
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        features, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)

    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")


    # 4. Train Decision Tree Model (Pipeline Step 6: Model Training)
    print("Training Decision Tree Classifier...")
    # Use your wrapper class
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train_encoded) # Wrapper class handles encoding internally if needed
    print("Decision Tree Classifier training complete.")


    # 5. Evaluate Model (Pipeline Step 7: Model Evaluation)
    print("Evaluating model performance...")
    # Use the predict method of your wrapper class
    y_pred_encoded = dt_model.predict(X_test) # Wrapper class returns original labels or encoded if fit_transform was used internally

    # If your wrapper's predict returns original labels, use y_test directly
    # If it returns encoded labels, compare with y_test_encoded
    # Assuming your wrapper's predict returns original labels based on its definition:
    y_test_original_labels = label_encoder.inverse_transform(y_test_encoded)
    accuracy = accuracy_score(y_test_original_labels, y_pred_encoded)

    print(f"Validation Accuracy: {accuracy*100:.2f}%")


    # 6. Save Model and Label Encoder (Pipeline Step 8: Model Saving)
    print(f"Saving model to {MODEL_PATH} and encoder to {ENCODER_PATH}...")
    # Save the internal scikit-learn model from the wrapper
    joblib.dump(dt_model.model, MODEL_PATH)
    # Save the label encoder used in the wrapper
    joblib.dump(dt_model.label_encoder, ENCODER_PATH)
    print("Model and encoder saved.")

# --- WebSocket Handler ---
async def websocket_handler(request):
    """Handles incoming websocket connections and image data."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print("WebSocket client connected.")

    # Load model and encoder if not loaded (e.g., if server restarted)
    global dt_model, label_encoder, classifiers
    if dt_model is None or label_encoder is None:
        print("Loading model and encoder for WebSocket...")
        try:
            # Load the raw scikit-learn model and the label encoder
            loaded_skl_model = joblib.load(MODEL_PATH)
            loaded_label_encoder = joblib.load(ENCODER_PATH)

            # Recreate the wrapper instance and assign the loaded components
            # Note: This assumes the wrapper class is defined and available
            dt_model = DecisionTreeClassifier() # Instantiate the wrapper
            dt_model.model = loaded_skl_model # Assign the loaded scikit-learn model
            dt_model.label_encoder = loaded_label_encoder # Assign the loaded label encoder

            print("Model and encoder loaded successfully.")
        except FileNotFoundError:
            print("Error: Model or encoder files not found. Run training first.")
            await ws.send_json({"error": "Model not trained. Please train the model first."})
            await ws.close()
            return
        except Exception as e:
             print(f"Error loading model or encoder: {e}")
             await ws.send_json({"error": f"Error loading model: {e}"})
             await ws.close()
             return


    if classifiers is None:
         print("Initializing Haar cascades for WebSocket...")
         # Update the path here as well for the websocket handler
         HAAR_CASCADES_DIR_WS = os.path.join(os.path.dirname(__file__), '..', 'haar-cascades')
        
         if 'haar_detector' in globals() and hasattr(haar_detector, 'HAAR_CASCADES_DIR'):
             original_haar_path = haar_detector.HAAR_CASCADES_DIR # Save original path
             haar_detector.HAAR_CASCADES_DIR = HAAR_CASCADES_DIR_WS # Temporarily set global path in the module
             classifiers = initialize_haar_detectors()
             haar_detector.HAAR_CASCADES_DIR = original_haar_path # Restore original path
         else:
             # Fallback if the haar_detector module structure is different
             print("Warning: Could not access haar_detector.HAAR_CASCADES_DIR. Attempting initialization with default path.")
             classifiers = initialize_haar_detectors()


         if classifiers is None:
              print("Error: Failed to load Haar cascades.")
              await ws.send_json({"error": "Failed to initialize face detector."})
              await ws.close()
              return


    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            message = msg.text
            # Expect base64 image string from client in format like:
            # "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
            try:
                header, encoded_img = message.split(",", 1)
                img_bytes = base64.b64decode(encoded_img)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decoding image data: {e}")
                await ws.send_json({"error": "Invalid image data"})
                continue

            if img is None:
                print("Failed to decode image.")
                await ws.send_json({"error": "Failed to decode image"})
                continue

            print("Received image, processing...")

            try:
                # --- Prediction Pipeline ---
                # The image received via websocket is already a numpy array (BGR)
                # Rotation correction (if needed, depends on source - generally mobile cameras add EXIF)
                # This function expects a path, might need adaptation or check if needed for websocket source
                # For simplicity, we'll assume the image is already upright from the client or handle it if needed.
                # If rotation is expected, you might need to save the image temporarily or adapt correct_image_rotation

                # Apply smoothing based on noise
                img_processed = apply_thresholded_smoothing(img)
                if img_processed is None:
                    await ws.send_json({"error": "Image processing failed (smoothing)"})
                    continue

                # Detect faces and eyes
                faces = detect_faces_and_eyes(img_processed, classifiers)
                if not faces:
                    print("No face detected in received image.")
                    await ws.send_json({"result": "No face detected"})
                    continue

                # Crop first face and resize
                x, y, w, h = faces[0]['rect']
                cropped_face = img_processed[y:y+h, x:x+w]
                if cropped_face.size == 0:
                    print("Cropped face is empty.")
                    await ws.send_json({"error": "Could not crop face region"})
                    continue

                resized_face = cv2.resize(cropped_face, IMAGE_SIZE)

                 # Convert to grayscale for feature extraction if not already
                if len(resized_face.shape) == 3 and resized_face.shape[-1] == 3:
                     resized_face_gray = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
                else:
                     resized_face_gray = resized_face # Already grayscale


                # Extract features using your extractor
                fe = FeatureExtractor(resized_face_gray)
                feature_vector = fe.calculate(visulalize=False).reshape(1, -1)

                # Predict using trained model (using the wrapper's predict method)
                pred_label = dt_model.predict(feature_vector)[0] # Predict returns array, get the first element

                print(f"Prediction: {pred_label}")
                # Send prediction back as JSON
                await ws.send_json({"predicted_person": pred_label})

            except Exception as e:
                print(f"Error during prediction pipeline: {e}")
                await ws.send_json({"error": f"Prediction failed: {e}"})

        elif msg.type == web.WSMsgType.ERROR:
            print('WebSocket connection closed with exception %s' % ws.exception())

    print('WebSocket connection closed.')
    return ws


# --- HTTP Handler ---
async def http_handler(request):
    """Serves a simple webpage or info."""
    return web.Response(text="<html><h1>Kabeer Face Recognition Server</h1><p>Connect via WebSocket on port 8765 for predictions.</p></html>", content_type='text/html')


async def start_servers():
    """Starts the HTTP and WebSocket servers."""
    # --- AioHTTP Server Setup ---
    app = web.Application()
    # Add WebSocket route
    app.router.add_get('/ws', websocket_handler)
    # Add simple HTTP route
    app.router.add_get('/', http_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    # Use 0.0.0.0 to bind to all network interfaces, allowing external access in Colab
    http_site = web.TCPSite(runner, '0.0.0.0', 8080)
    websocket_site = web.TCPSite(runner, '0.0.0.0', 8765) # Separate port for WebSocket

    print("Starting HTTP server on http://0.0.0.0:8080")
    await http_site.start()

    print("Starting WebSocket server on ws://0.0.0.0:8765")
    await websocket_site.start()

    # Keep the server running
    print("Servers are running. Press Ctrl+C to stop.")
    await asyncio.Future()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # First, train the model
        train_face_recognition_model()
        # Then, start the servers
        asyncio.run(start_servers())
    except KeyboardInterrupt:
        print("\nServers stopped by user.")
    except Exception as e:
        print(f"An error occurred during setup or runtime: {e}")