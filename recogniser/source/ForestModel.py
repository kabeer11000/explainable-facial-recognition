import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class RandomForestClassifier:
    """
    A wrapper class for scikit-learn's RandomForestClassifier.
    This provides a simplified interface for fixed-size feature vectors
    and classification of people labels.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=42):
        """
        Initializes the Random Forest Classifier using scikit-learn's implementation.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int, optional): The maximum depth of the tree. If None, the tree
                                       will grow until all leaves are pure or
                                       min_samples_leaf is reached.
            min_samples_split (int): The minimum number of samples required to split an
                                     internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a
                                    leaf node.
            criterion (str): The function to measure the quality of a split.
                             Supported criteria are "gini" for the Gini impurity
                             and "entropy" for the information gain.
            random_state (int, optional): Controls the randomness of the estimator.
                                          Useful for reproducibility.
        """
        self.model = SklearnRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder() # To handle non-numeric labels
        print("Random Forest Classifier (Scikit-learn) initialized.")

    def fit(self, X, y):
        """
        Trains the Random Forest Classifier.

        Args:
            X (np.ndarray): Training feature data, shape (n_samples, n_features).
                            Expected to be a fixed-size vector for each sample.
            y (np.ndarray): Training labels, shape (n_samples,).
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be NumPy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array (n_samples,).")

        print("Starting to fit Random Forest Classifier (Scikit-learn)...")
        # Encode labels if they are not numeric (e.g., strings)
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        print("Random Forest Classifier (Scikit-learn) fitting complete.")

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X (np.ndarray): Feature data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels, shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Classifier not fitted. Call .fit(X, y) first.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        
        print(f"Predicting labels for {X.shape[0]} samples...")
        predictions_encoded = self.model.predict(X)
        # Decode numeric predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        print("Prediction complete.")
        return predictions

    def predict_proba(self, X):
        """
        Predicts class probabilities for new data.

        Args:
            X (np.ndarray): Feature data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, n_classes).
                        The columns are in the order of the classes known by the encoder.
        """
        if self.model is None:
            raise RuntimeError("Classifier not fitted. Call .fit(X, y) first.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        
        print(f"Predicting probabilities for {X.shape[0]} samples...")
        probabilities = self.model.predict_proba(X)
        print("Probability prediction complete.")
        return probabilities
