import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Contrast stretching
min_val = np.min(image)
max_val = np.max(image)
contrast_stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Median filtering
median_filtered = cv2.medianBlur(image, 3)

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Contrast Stretched')
plt.imshow(contrast_stretched, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Median Filtered')
plt.imshow(median_filtered, cmap='gray')

plt.show()
