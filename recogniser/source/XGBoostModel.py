import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb # Import XGBoost
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving/loading models
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

# --- Your original code adapted for XGBoost ---

# Assuming these imports are from your project structure
# from feature_extractor import FeatureExtractor
# from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
# from utilities import apply_thresholded_smoothing, estimate_noise_level

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
if num_classes > 2:
    xgb_objective = 'multi:softmax'
    xgb_eval_metric = 'mlogloss' # Multi-class logloss
    print(f"Detected {num_classes} classes. Using objective='multi:softmax' and eval_metric='mlogloss'.")
    model = xgb.XGBClassifier(
        objective=xgb_objective,
        num_class=num_classes, # Required for multi:softmax
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric=xgb_eval_metric,
        random_state=42
    )
else: # Binary classification
    xgb_objective = 'binary:logistic'
    xgb_eval_metric = 'logloss'
    print(f"Detected {num_classes} classes. Using objective='binary:logistic' and eval_metric='logloss'.")
    model = xgb.XGBClassifier(
        objective=xgb_objective,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric=xgb_eval_metric,
        random_state=42
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

# --- Original sample data generation and hyperparameter tuning sections (kept for reference/completeness) ---

# --- 1. Generate Sample Data ---
# For demonstration, let's create a synthetic dataset.
# In a real scenario, you would load your data using pd.read_csv(), etc.
# print("Generating synthetic data...")
# np.random.seed(42) # for reproducibility

# Number of samples
# n_samples = 1000
# Number of features
# n_features = 10

# Create random features
# X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])

# Create a target variable (binary classification)
# Let's make it somewhat dependent on a few features
# y = ((X['feature_0'] * 2 + X['feature_3'] * 0.5 - X['feature_7'] * 1.5 + np.random.randn(n_samples) * 0.5) > 1.0).astype(int)

# print(f"Data generated: X shape {X.shape}, y shape {y.shape}")
# print("First 5 rows of X:")
# print(X.head())
# print("\nFirst 5 values of y:")
# print(y.head())
# print(f"\nClass distribution in y:\n{y.value_counts()}")

# --- 2. Split Data into Training and Testing Sets ---
# This is crucial for evaluating how well your model generalizes to unseen data.
# print("\nSplitting data into training and testing sets...")
# X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print(f"X_train shape: {X_train_synth.shape}, y_train shape: {y_train_synth.shape}")
# print(f"X_test shape: {X_test_synth.shape}, y_test shape: {y_test_synth.shape}")

# --- 3. Initialize and Train the XGBoost Model (Synthetic Data Example) ---
# This section is commented out as it's superseded by your specific code.
# print("\nInitializing and training XGBoost Classifier (Synthetic Data Example)...")
# model_synth = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )
# model_synth.fit(X_train_synth, y_train_synth)
# print("XGBoost model training complete (Synthetic Data Example).")

# --- 4. Make Predictions (Synthetic Data Example) ---
# y_pred_synth = model_synth.predict(X_test_synth)
# y_pred_proba_synth = model_synth.predict_proba(X_test_synth)[:, 1]

# --- 5. Evaluate the Model's Performance (Synthetic Data Example) ---
# print("\nEvaluating model performance (Synthetic Data Example):")
# accuracy_synth = accuracy_score(y_test_synth, y_pred_synth)
# print(f"Accuracy: {accuracy_synth:.4f}")
# print("\nClassification Report (Synthetic Data Example):")
# print(classification_report(y_test_synth, y_pred_synth))
# cm_synth = confusion_matrix(y_test_synth, y_pred_synth)
# print("\nConfusion Matrix (Synthetic Data Example):")
# print(cm_synth)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_synth, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Predicted 0', 'Predicted 1'],
#             yticklabels=['Actual 0', 'Actual 1'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix for XGBoost Classifier (Synthetic Data Example)')
# plt.show()

# print("\nFeature Importances (Top 5 - Synthetic Data Example):")
# feature_importances_synth = pd.Series(model_synth.feature_importances_, index=X_train_synth.columns).sort_values(ascending=False)
# print(feature_importances_synth.head())
# plt.figure(figsize=(10, 6))
# sns.barplot(x=feature_importances_synth.head(10).values, y=feature_importances_synth.head(10).index, palette='viridis')
# plt.title('Top 10 Feature Importances (XGBoost - Synthetic Data Example)')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()

# --- Optional: Cross-validation with XGBoost (Synthetic Data Example) ---
# from sklearn.model_selection import cross_val_score
# print("\nPerforming 5-fold cross-validation (Synthetic Data Example)...")
# cv_scores_synth = cross_val_score(model_synth, X, y, cv=5, scoring='accuracy')
# print(f"Cross-validation accuracies (Synthetic Data Example): {cv_scores_synth}")
# print(f"Mean CV accuracy (Synthetic Data Example): {cv_scores_synth.mean():.4f}")
# print(f"Standard deviation of CV accuracy (Synthetic Data Example): {cv_scores_synth.std():.4f}")

# --- Optional: Hyperparameter Tuning with GridSearchCV (Synthetic Data Example) ---
# from sklearn.model_selection import GridSearchCV
# print("\nPerforming GridSearchCV for hyperparameter tuning (Synthetic Data Example - this might take a moment)...")
# param_grid_synth = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8, 0.9]
# }
# grid_search_synth = GridSearchCV(
#     estimator=xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42),
#     param_grid=param_grid_synth,
#     scoring='accuracy',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )
# grid_search_synth.fit(X_train_synth, y_train_synth)
# print(f"\nBest parameters found by GridSearchCV (Synthetic Data Example): {grid_search_synth.best_params_}")
# print(f"Best cross-validation accuracy (Synthetic Data Example): {grid_search_synth.best_score_:.4f}")
# best_model_synth = grid_search_synth.best_estimator_
# y_pred_best_synth = best_model_synth.predict(X_test_synth)
# accuracy_best_synth = accuracy_score(y_test_synth, y_pred_best_synth)
# print(f"Accuracy with best model on test set (Synthetic Data Example): {accuracy_best_synth:.4f}")
