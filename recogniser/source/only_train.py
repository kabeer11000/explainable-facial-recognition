import os
from pathlib import Path
from feature_extractor import FeatureExtractor
from haar_detector import initialize_haar_detectors, detect_faces_and_eyes
from utilities import apply_thresholded_smoothing, estimate_noise_level
from Model import DecisionTreeClassifier
import joblib
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Before training
features = np.array([])
labels = np.array([])

features = joblib.load('features.pkl')
labels = joblib.load('labels.pkl')

print("Features and labels dumped successfully using joblib.")

from sklearn.model_selection import train_test_split

# X = features, y = labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


print("Training Decision Tree Classifier...")
clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=8)
clf.fit(X_train, y_train)
print("Training complete.")

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report

print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Save the trained model
joblib.dump(clf, "decision_tree_model.pkl")
joblib.dump(clf.label_encoder, "label_encoder.pkl")
print("Model saved as decision_tree_model.pkl")
