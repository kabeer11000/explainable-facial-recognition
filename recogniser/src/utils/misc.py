# Define Utility functions (Noise estimation, Smoothing, Rotation)
import cv2
import numpy as np
from PIL import Image, ExifTags

def estimate_noise_level(image_np: np.ndarray) -> float:
    """
    Estimates the noise level of an image using the Laplacian variance method.

    Args:
        image_np (np.ndarray): The input image as a NumPy array.

    Returns:
        float: The estimated noise level (variance of the Laplacian).
    """
    if len(image_np.shape) == 3:
        # Convert to grayscale if the image is in color
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_np

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    # Compute the variance of the Laplacian
    noise_level = laplacian.var()
    return noise_level

def apply_thresholded_smoothing(image_np: np.ndarray, noise_threshold: float = 700.0, smoothing_kernel_size: tuple = (5, 5)) -> np.ndarray:
    """
    Estimates the noise level of an image and applies Gaussian smoothing if it's too noisy.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (BGR format).
        noise_threshold (float): The variance of Laplacian threshold above which
                                 an image is considered "too noisy" and will be smoothed.
        smoothing_kernel_size (tuple): The kernel size for Gaussian blur (e.g., (5,5)).
                                         Must be odd and positive.

    Returns:
        np.ndarray: The smoothed image if noise is detected, otherwise the original image.
    """
    if image_np is None:
        print("Error: Input image is None for apply_gaussian.")
        return None

    # Estimate noise level
    noise_level = estimate_noise_level(image_np)

    print(f"Smoothen: Estimated noise level: {noise_level:.2f}")

    if noise_level > noise_threshold:
        print(f"Smoothen: Image is noisy (>{noise_threshold}). Applying Gaussian blur.")
        # Apply Gaussian Blur
        smoothed_image = cv2.GaussianBlur(image_np, smoothing_kernel_size, 0)
        return smoothed_image
    else:
        print(f"Smoothen: Image is clean enough (<={noise_threshold}). Returning original.")
        return image_np

def rotate_pillow_image_from_exif(image: Image):
    """
    Reads the EXIF Orientation tag from a PIL Image and rotates the image
    accordingly. If no EXIF data or Orientation tag, returns original.

    Args:
        image (PIL.Image.Image): The input PIL Image object.

    Returns:
        PIL.Image.Image: The rotated (or original if no rotation needed) PIL Image object.
    """
    img_exif = image.getexif()

    if img_exif is None:
        print(f"No EXIF data found for Image. Returning original image.")
        return image

    orientation = 1 # Default orientation (no rotation)

    # Find the 'Orientation' tag value
    for tag_id, value in img_exif.items():
        if tag_id in ExifTags.TAGS and ExifTags.TAGS[tag_id] == 'Orientation':
            orientation = value
            break
        elif tag_id == 0x0112: # Numeric ID for Orientation tag
            orientation = value
            break

    # Apply rotation based on Orientation tag
    if orientation == 3: # Rotate 180 degrees
        print(f"Rotating by 180 degrees (EXIF: {orientation}).")
        image = image.rotate(180, expand=True)
    elif orientation == 6: # Rotate 90 degrees CW
        print(f"Rotating by 90 degrees CW (EXIF: {orientation}).")
        image = image.rotate(-90, expand=True) # PIL rotate() is CCW, so -90 for CW
    elif orientation == 8: # Rotate 270 degrees CW (or 90 CCW)
        print(f"Rotating by 270 degrees CW (EXIF: {orientation}).")
        image = image.rotate(90, expand=True) # PIL rotate() is CCW, so 90 for 270 CW
    elif orientation != 1:
        print(f"Applying auto-orientation for (EXIF: {orientation}).")

    return image
