import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filters
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# Combine the results
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Sobel X')
plt.imshow(cv2.convertScaleAbs(sobel_x), cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Sobel Y')
plt.imshow(cv2.convertScaleAbs(sobel_y), cmap='gray')
plt.show()