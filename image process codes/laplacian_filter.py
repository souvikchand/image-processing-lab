import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Laplacian filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Laplacian Filtered Image')
plt.imshow(laplacian, cmap='gray')
plt.show()