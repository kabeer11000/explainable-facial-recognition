import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier
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
    print("Features and labels loaded successfully using joblib.")
    print(f"Loaded features shape: {features.shape}, labels shape: {labels.shape}")
except FileNotFoundError:
    print("Error: 'features.pkl' or 'labels.pkl' not found. Please ensure they exist.")
    # Exit or handle the error appropriately if files are missing
    exit() # For demonstration, exiting if files are not found

# Ensure labels are in a format compatible with scikit-learn (typically integers 0, 1, ... for classification)
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


print("\nTraining Decision Tree Classifier...")
# Initialize Decision Tree Classifier
# Key parameters:
# - max_depth: The maximum depth of the tree. Controls overfitting.
# - random_state: For reproducibility.
# - criterion: The function to measure the quality of a split ('gini' for Gini impurity, 'entropy' for information gain).
# - min_samples_split: The minimum number of samples required to split an internal node.
# - min_samples_leaf: The minimum number of samples required to be at a leaf node.

model = DecisionTreeClassifier(
    max_depth=10,          # A reasonable starting point; tune this value
    random_state=42,       # For reproducibility
    criterion='gini',      # Gini impurity is default and often works well
    # min_samples_split=2,   # Default: 2
    # min_samples_leaf=1,    # Default: 1
    # class_weight='balanced' # Consider using this if your classes are imbalanced
)

# Train the model
model.fit(X_train, y_train)
print("Decision Tree model training complete.")

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
    plt.title('Confusion Matrix for Decision Tree Classifier') # Changed title
    plt.show()
except Exception as e:
    print(f"Could not plot Confusion Matrix: {e}. Ensure matplotlib and seaborn are installed.")


# Save the trained model
joblib.dump(model, "decision_tree_model.pkl") # Changed filename
print("Model saved as decision_tree_model.pkl")