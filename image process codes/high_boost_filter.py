import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# Scaling factor
k = 1.5
# Blur the image
blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
# High-boost filtering
high_boosted_image = cv2.addWeighted(image, k, blurred, 1 - k, 0)
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('High-Boost Filtering')
plt.imshow(high_boosted_image, cmap='gray')
plt.show()