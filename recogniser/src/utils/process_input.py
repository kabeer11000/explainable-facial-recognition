from skimage.transform import resize
import cv2

# Resizes and grayscales the image, returns both colored and grayscaled
def process_input_image(frame, IMAGE_SIZE): 
    # Convert frame to grayscale before resizing if your model expects grayscale
    # Assuming your `cropped_face_gray` variable name indicates grayscale expectation
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale frame
    # The `resize` function from skimage.transform returns float64 in [0.0, 1.0]
    resized_frame_gray_float = resize(frame_gray, IMAGE_SIZE, anti_aliasing=True)
    resized_frame_float = resize(frame, IMAGE_SIZE, anti_aliasing=True)
    
    return resized_frame_gray_float, resized_frame_float

