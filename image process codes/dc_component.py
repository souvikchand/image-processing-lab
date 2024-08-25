import cv2
import numpy as np

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the DC component (average intensity)
dc_component = np.mean(image)

print(f"DC Component (Average Intensity): {dc_component}")
