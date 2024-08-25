import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply median filter
median_filtered = cv2.medianBlur(image, 5) # Kernel size of 5x5
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Median Filtered Image')
plt.imshow(median_filtered, cmap='gray')
plt.show()