import joblib

def load_model(MODEL_PATH="saved_models/xgboost_model.pkl", LABEL_ENCODER_PATH="saved_models/label_encoder.pkl"):
    try:
        loaded_model = joblib.load(MODEL_PATH)
        print("XGBoost model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{MODEL_PATH}' not found. Please train and save the model first.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    try:
        loaded_le = joblib.load(LABEL_ENCODER_PATH)
        print("LabelEncoder loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{LABEL_ENCODER_PATH}' not found.")
        exit()
    except Exception as e:
        print(f"Error loading LabelEncoder: {e}")
        exit()

    print(f"Model type: {type(loaded_model)}")
    print(f"LabelEncoder classes: {loaded_le.classes_}")

    return loaded_model, loaded_le
