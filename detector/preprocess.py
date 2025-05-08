import cv2
import numpy as np

def preprocess_input(image):
    # Resize the image to the shape your model expects
    image_resized = cv2.resize(image, (100, 100))  # adjust (100,100) as needed
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    flattened = normalized.flatten()
    return flattened