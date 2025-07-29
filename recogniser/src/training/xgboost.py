import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb # Import XGBoost
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving/loading models
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

# Before training
features = np.array([])
labels = np.array([])

# Load features and labels from joblib files
try:
    features = joblib.load('features.pkl')
    labels = joblib.load('labels.pkl')
    print("Features and labels dumped successfully using joblib.")
    print(f"Loaded features shape: {features.shape}, labels shape: {labels.shape}")
except FileNotFoundError:
    print("Error: 'features.pkl' or 'labels.pkl' not found. Please ensure they exist.")
    # Exit or handle the error appropriately if files are missing
    exit() # For demonstration, exiting if files are not found

# Ensure labels are in a format compatible with XGBoost (typically integers 0, 1, ... for classification)
# If your labels are strings or non-integer numerical labels, encode them using LabelEncoder.
print("\nEncoding labels using sklearn.preprocessing.LabelEncoder...")
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print(f"Original labels unique values: {np.unique(labels)}")
print(f"Encoded labels unique values: {np.unique(labels_encoded)}")

# Save the fitted LabelEncoder for later use (e.g., to decode predictions)
joblib.dump(le, "label_encoder.pkl")
print("LabelEncoder saved as label_encoder.pkl")


# Split data into training and testing sets
# X = features, y = labels_encoded
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)


print("\nTraining XGBoost Classifier...")
# Initialize XGBoost Classifier
# Key parameters for classification:
# - objective: 'binary:logistic' for binary classification, 'multi:softmax' for multi-class (requires num_class)
# - n_estimators: Number of boosting rounds (trees)
# - learning_rate: Step size shrinkage
# - max_depth: Maximum depth of a tree
# - eval_metric: Metric used for early stopping (if validation set is provided during fit)

# Determine the objective based on the number of unique classes
num_classes = len(le.classes_)
xgb_objective = 'multi:softmax'
xgb_eval_metric = 'mlogloss' # Multi-class logloss
print(f"Detected {num_classes} classes. Using objective='multi:softmax' and eval_metric='mlogloss'.")
model = xgb.XGBClassifier(
  objective=xgb_objective,
  num_class=num_classes, # Required for multi:softmax
  n_estimators=200,
  learning_rate=0.2,
  max_depth=8,
  eval_metric=xgb_eval_metric
)

# Train the model
model.fit(X_train, y_train)
print("XGBoost model training complete.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\nEvaluating model performance:")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
# Use target_names for better readability if original labels are strings
target_names = le.classes_
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Visualize Confusion Matrix (if matplotlib and seaborn are available)
try:
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for XGBoost Classifier')
    plt.show()
except Exception as e:
    print(f"Could not plot Confusion Matrix: {e}. Ensure matplotlib and seaborn are installed.")


# Save the trained model
joblib.dump(model, "xgboost_model.pkl")
print("Model saved as xgboost_model.pkl")
