import asyncio
import websockets
import json
import base64
import cv2  # Used only for imdecode/imencode
import numpy as np
from skimage.transform import resize # <--- ADD THIS IMPORT for 'resize'

import joblib
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes # Not used in this snippet but keep if needed elsewhere
from utilities import apply_thresholded_smoothing # Not used in this snippet but keep if needed elsewhere
import xgboost as xgb # Import XGBoost

# --- Configuration ---
IMAGE_SIZE = (128, 128) 
MODEL_PATH = 'xgboost_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
# Initialize an instance of your custom DecisionTreeClassifier wrapper
# RECOGNITION_MODEL = xgb.XGBClassifier(
#     objective='binary:logistic',  # Or 'multi:softmax' if you have more than two classes
#     n_estimators=100,             # Similar to n_estimators in RandomForest
#     learning_rate=0.1,            # Controls the step size shrinkage
#     max_depth=5,                  # Controls the complexity of the trees
#     use_label_encoder=False,      # Recommended for newer XGBoost versions
#     eval_metric='logloss',        # A common evaluation metric for classification
#     random_state=42               # For reproducibility
# )
def setup_model():
    print("Loading model for WebSocket...")
    try:
        # Load the raw scikit-learn model
        loaded_skl_model = joblib.load(MODEL_PATH)
        
        # Assign the loaded scikit-learn model to your wrapper's 'model' attribute
        RECOGNITION_MODEL = loaded_skl_model

        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Run training first.")
        # It's good practice to exit if a critical resource isn't found
        exit(1) 
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1) # Exit if there's a critical error
    print("--- Server is ready ---")

def base64_to_image(base64_string: str) -> np.ndarray:
    """ Converts a Base64 encoded string to a CV2 image. """
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        # IMREAD_COLOR reads as BGR, which is OpenCV's default
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from base64 data.")
        return img
    except Exception as e:
        print(f"Error decoding base64 to image: {e}")
        return None # Return None or raise the error

async def websocket_handler(websocket):
    """ Main event handler for WebSocket communication. """
    # print(f"Client connected from path: {path}") # Path argument is now used
    try:
        async for message in websocket:
            request = json.loads(message)
            event, request_id = request.get("event"), request.get("id")

            if event != "image_push":
                print(f"Received unknown event: {event}")
                continue

            frame = base64_to_image(request.get("data"))
            if frame is None:
                response = {"event": f"server_response", "status": "error", "message": "Failed to decode image."}
                await websocket.send(json.dumps(response))
                continue

            # Convert frame to grayscale before resizing if your model expects grayscale
            # Assuming your `cropped_face_gray` variable name indicates grayscale expectation
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale frame
            # The `resize` function from skimage.transform returns float64 in [0.0, 1.0]
            resized_frame_gray_float = resize(frame_gray, IMAGE_SIZE, anti_aliasing=True)
            resized_frame_float = resize(frame, IMAGE_SIZE, anti_aliasing=True)
            
            # Convert back to uint8 [0-255] for OpenCV operations like imshow or saving
            cropped_face_gray_uint8 = (resized_frame_gray_float * 255).astype(np.uint8)

            # Display the image for debugging (will block the server if not handled carefully)
            # You might want to remove or make this conditional for a production server
            # cv2.imshow('Processed Frame (Grayscale)', cropped_face_gray_uint8)
            # cv2.waitKey(1) # Display for 1 ms and continue, preventing blocking

            # 5. Extract features - ensure FeatureExtractor expects float64 [0.0, 1.0] 
            # or uint8 [0-255] based on its internal implementation.
            # If it expects float, pass resized_frame_gray_float
            # If it expects uint8, pass cropped_face_gray_uint8
            # Assuming FeatureExtractor expects the normalized float image based on typical ML pipelines
            extractor = FeatureExtractor(resized_frame_gray_float)
            feature_vector = np.concatenate([extractor.calculate().flatten(), resized_frame_float.flatten()])

            prediction_input = np.array([feature_vector]) # Wrap in a list for single sample prediction
            
            # Ensure the model is loaded before trying to predict
            if RECOGNITION_MODEL is None:
                 print("Recognition model is not loaded. Cannot predict.")
                 response = {"event": f"server_response", "status": "error", "message": "Model not ready."}
                 await websocket.send(json.dumps(response))
                 continue

            predicted_label = RECOGNITION_MODEL.predict(prediction_input)[0]
            predicted_probabilty = RECOGNITION_MODEL.predict_proba(prediction_input)[0]
            print(f"[{request_id}] Recognition complete. Label: {predicted_label} and Confidence: {predicted_probabilty}")

            response = {
                "event": f"server_response", 
                "status": "recognized", 
                "label": (predicted_label), # if (predicted_probabilty[int(predicted_label)-1]) > 0.5 else 'Unknown', 
                "confidence": predicted_probabilty.tolist()
            }
            print(response)
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected gracefully.")
    except json.JSONDecodeError:
        print("Received invalid JSON message.")
    except Exception as e:
        print(f"An unexpected error occurred in websocket_handler: {e}")
        # Optionally send an error response to the client
        try:
            if websocket:
                response = {"event": "server_error", "message": str(e)}
                await websocket.send(json.dumps(response))
        except Exception as send_e:
            print(f"Error sending error message: {send_e}")


async def main():
    setup_model()
    cors_headers = {
        "Access-Control-Allow-Origin": "*",  # Or specify your allowed origins, e.g., "http://localhost:3000"
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS", # Important for preflight if it were an HTTP endpoint
        "Access-Control-Allow-Headers": "Content-Type, Authorization", # Relevant headers
        "Access-Control-Max-Age": "86400", # Cache preflight response for 24 hours
    }

    # The `websockets.serve` function correctly passes `websocket` and `path`
    # to the `websocket_handler` function.
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future() # Keep the server running indefinitely

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by KeyboardInterrupt.")
    except Exception as e:
        print(f"An error occurred during server startup: {e}")
