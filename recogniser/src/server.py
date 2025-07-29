import asyncio
import websockets
import json
import base64
import cv2  
import numpy as np
import joblib

from utils.process_input import process_input_image
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes # Not used in this snippet but keep if needed elsewhere
from utils.misc import apply_thresholded_smoothing # Not used in this snippet but keep if needed elsewhere

# --- Configuration ---
IMAGE_SIZE = (128, 128) 

from setup.decisiontree import load_model as load_model_decisiontree
MODEL_PATH_1 = 'saved_models/decision_tree_model.pkl'

from setup.randomforest import load_model as load_model_randomforest
MODEL_PATH_2 = 'saved_models/random_forest_model.pkl'

from setup.xgboost import load_model as load_model_xgboost
MODEL_PATH_3 = 'saved_models/xgboost_model.pkl'

LABEL_ENCODER_PATH = 'saved_models/12-dataset/label_encoder.pkl'

print("Loading model for WebSocket...")
try:
    RECOGNITION_MODEL_1, LABEL_ENCODER = load_model_decisiontree(MODEL_PATH_1, LABEL_ENCODER_PATH)
    RECOGNITION_MODEL_2, LABEL_ENCODER = load_model_randomforest(MODEL_PATH_2, LABEL_ENCODER_PATH)
    RECOGNITION_MODEL_3, LABEL_ENCODER = load_model_xgboost(MODEL_PATH_3, LABEL_ENCODER_PATH)
    print("Models loaded successfully.")
except FileNotFoundError:
    print(f"Error: Models file '{MODEL_PATH_1}' not found. Run training first.")
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
    try:
        async for message in websocket:
            request = json.loads(message)
            event, request_id, model = request.get("event"), request.get("id"), request.get('model')

            if event != "image_push":
                print(f"Received unknown event: {event}")
                continue

            frame = base64_to_image(request.get("data"))
            if frame is None:
                response = {"event": f"server_response", "status": "error", "message": "Failed to decode image."}
                await websocket.send(json.dumps(response))
                continue

            # 1. Preprocess the input frame
            resized_frame_gray_uint8, _ = process_input_image(frame, IMAGE_SIZE)

            extractor = FeatureExtractor(resized_frame_gray_uint8) # Pass the uint8 image
            feature_vector = extractor.calculate()

            prediction_input = feature_vector.reshape(1, -1) # More concise than np.array([feature_vector])

            # Use xgboost by default
            if model is None: model = 'xgboost'

            if (model == 'decisiontree'):
                predicted_label_encoded_np = RECOGNITION_MODEL_1.predict(prediction_input)[0] # This will be np.int64
                predicted_probabilities_np = RECOGNITION_MODEL_1.predict_proba(prediction_input)[0] # This will be np.ndarray of floats
            if (model == 'randomforest'):
                predicted_label_encoded_np = RECOGNITION_MODEL_2.predict(prediction_input)[0] # This will be np.int64
                predicted_probabilities_np = RECOGNITION_MODEL_2.predict_proba(prediction_input)[0] # This will be np.ndarray of floats
            if (model == 'xgboost'):
                predicted_label_encoded_np = RECOGNITION_MODEL_3.predict(prediction_input)[0] # This will be np.int64
                predicted_probabilities_np = RECOGNITION_MODEL_3.predict_proba(prediction_input)[0] # This will be np.ndarray of floats

            confidence_for_predicted_label = predicted_probabilities_np[predicted_label_encoded_np]

            # 6. Apply thresholding and decode the label
            threshold = 0.5 # Tune this value based on your requirements
            if confidence_for_predicted_label > threshold:
                final_label = LABEL_ENCODER.inverse_transform([predicted_label_encoded_np])[0]
            else:
                final_label = 'Unknown'

            # 7. Prepare the response for JSON serialization
            response = {
                "event": "server_response",
                "status": "recognized",
                "model": model,
                "label": str(final_label), # Ensure it's a string
                "original_label": str(LABEL_ENCODER.inverse_transform([predicted_label_encoded_np])[0]),
                "untransformed_label": int(predicted_label_encoded_np), # Convert np.int64 to int
                "confidence": float(confidence_for_predicted_label) # Convert np.float64 to float
            }

            # 8. Print and send response
            print(f"[{request_id}] Recognition complete. Label: {final_label} and Confidence: {confidence_for_predicted_label:.4f} and all Probabilities: {predicted_probabilities_np}")
            print(response) # This will now print the JSON-serializable dictionary
            await websocket.send(json.dumps(response))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected gracefully.")
    except json.JSONDecodeError:
        print("Received invalid JSON message.")
    except Exception as e:
        print(f"An unexpected error occurred in websocket_handler: {e}")
        try:
            if websocket:
                response = {"event": "server_error", "message": str(e)}
                await websocket.send(json.dumps(response))
        except Exception as send_e:
            print(f"Error sending error message: {send_e}")


async def main():
    cors_headers = {
        "Access-Control-Allow-Origin": "*",  # Or specify your allowed origins, e.g., "http://localhost:3000"
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS", # Important for preflight if it were an HTTP endpoint
        "Access-Control-Allow-Headers": "Content-Type, Authorization", # Relevant headers
        "Access-Control-Max-Age": "86400", # Cache preflight response for 24 hours
    }

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
