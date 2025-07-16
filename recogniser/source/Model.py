import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class DecisionTreeClassifier:
    """
    A wrapper class for scikit-learn's Decision Tree Classifier.
    This provides a simplified interface for fixed-size feature vectors
    and classification of people labels.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=None):
        """
        Initializes the Decision Tree Classifier using scikit-learn's implementation.

        Args:
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
        self.model = SklearnDecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder() # To handle non-numeric labels
        print("Decision Tree Classifier (Scikit-learn) initialized.")

    def fit(self, X, y):
        """
        Trains the Decision Tree Classifier.

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

        print("Starting to fit Decision Tree Classifier (Scikit-learn)...")
        # Encode labels if they are not numeric (e.g., strings)
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        print("Decision Tree Classifier (Scikit-learn) fitting complete.")

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

# --- Example Usage ---
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    # Generate some synthetic data for demonstration
    # Features: [age, income, education_years]
    # Labels: ['buys_product_A', 'buys_product_B']

    # Sample data for people classification
    X = np.array([
        [30, 50000, 12],
        [25, 60000, 16],
        [45, 40000, 10],
        [35, 75000, 18],
        [28, 55000, 14],
        [50, 30000, 8],
        [40, 65000, 15],
        [22, 48000, 13],
        [55, 35000, 9],
        [33, 70000, 17],
        [29, 52000, 13],
        [48, 42000, 11],
        [31, 51000, 12], # Added more data for better splitting
        [26, 61000, 16],
        [44, 41000, 10],
        [36, 74000, 18]
    ])

    y = np.array([
        'buys_product_A',
        'buys_product_B',
        'buys_product_A',
        'buys_product_B',
        'buys_product_A',
        'buys_product_A',
        'buys_product_B',
        'buys_product_A',
        'buys_product_A',
        'buys_product_B',
        'buys_product_A',
        'buys_product_A',
        'buys_product_A',
        'buys_product_B',
        'buys_product_A',
        'buys_product_B'
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("--- Training with default parameters ---")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions_test = dt_classifier.predict(X_test)
    print("\nTest data predictions:")
    for i, sample in enumerate(X_test):
        print(f"Sample {sample}: Predicted = {predictions_test[i]}, Actual = {y_test[i]}")

    accuracy = accuracy_score(y_test, predictions_test)
    print(f"\nAccuracy on test set: {accuracy:.2f}")

    # Make predictions on new, unseen data
    X_new = np.array([
        [32, 58000, 14], # Should be A (similar to 30,50,12)
        [27, 62000, 17], # Should be B (similar to 25,60,16)
        [60, 25000, 7],  # Should be A (older, lower income)
        [38, 80000, 19]  # Should be B (higher income, education)
    ])

    predictions_new = dt_classifier.predict(X_new)
    print("\nNew data predictions:")
    for i, sample in enumerate(X_new):
        print(f"Sample {sample}: Predicted label = {predictions_new[i]}")

    print("\n--- Training with max_depth=2 and entropy criterion ---")
    dt_classifier_tuned = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=42)
    dt_classifier_tuned.fit(X_train, y_train)
    predictions_tuned_test = dt_classifier_tuned.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, predictions_tuned_test)
    print(f"\nAccuracy on test set (tuned): {accuracy_tuned:.2f}")
