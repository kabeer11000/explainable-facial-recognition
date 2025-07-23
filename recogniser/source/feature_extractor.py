# Define the FeatureExtractor class (HOG and LBP)
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    """
    A class to extract HOG and LBP features from a grayscale image.
    """

    def __init__(self, image: np.ndarray):
        """
        Initializes the FeatureExtractor with an image.

        Args:
            image (np.ndarray): Grayscale image, uint8 or float.
        """
        # Ensure the image is converted to uint8 if it's float,
        # as skimage functions generally expect uint8 for feature extraction.
        # This handles cases where cv2.imread might load as float if specified,
        # or if previous processing resulted in a float image.
        if image.dtype != np.uint8:
            self.image = img_as_ubyte(image)
        else:
            self.image = image

    def HOG(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute HOG features and return the visualization image.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - 1D array of HOG features.
                - 2D array of the HOG visualization image.
        """
        # Extract HOG features and get visualization image
        features, hog_image = hog(
            self.image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            block_norm='L2-Hys',
            # Add multichannel=False if your image might be mistakenly interpreted as multichannel
            # even if it's grayscale. For explicit grayscale, it's usually not needed.
        )
        return features, hog_image

    def LBP(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute LBP histogram and return the LBP image.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Normalized histogram of LBP patterns (1D array).
                - LBP image (2D array).
        """
        radius = 3
        n_points = 8 * radius

        # Compute Local Binary Pattern image
        lbp = local_binary_pattern(self.image, n_points, radius, method='uniform')

        # Calculate the histogram of LBP patterns
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_points + 3),
            range=(0, n_points + 2)
        )

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6) # Add a small epsilon to avoid division by zero

        return hist, lbp

    def calculate(self, visulalize:bool = False) -> np.ndarray:
        """
        Calculates and concatenates HOG and LBP features.
        Visualizes HOG features.

        Returns:
            np.ndarray: Concatenated HOG and LBP features.
        """
        hog_features, hog_image = self.HOG()
        lbp_features, lbp_image = self.LBP()

        if (visulalize):
            # Display HOG visualization image
            plt.figure(figsize=(4, 4))
            plt.title("HOG Visualization")
            plt.axis('off')
            plt.imshow(hog_image, cmap='gray')
            plt.show()

            # Display LBP image visualization
            plt.figure(figsize=(4, 4))
            plt.title("LBP Image")
            plt.axis('off')
            plt.imshow(lbp_image, cmap='gray')
            plt.show()

        # Concatenate HOG and LBP features into one vector (feature vector)
        return np.concatenate([hog_features, lbp_features])