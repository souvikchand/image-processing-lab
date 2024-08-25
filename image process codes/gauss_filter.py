import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian filter
gaussian_blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.5)
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Gaussian Blurred Image')
plt.imshow(gaussian_blurred, cmap='gray')
plt.show()