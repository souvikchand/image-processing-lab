import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# Logarithmic transformation
c = 255 / np.log(1 + np.max(image))
log_transformed = (c * np.log(1 + image)).astype(np.uint8)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Logarithmic Transformed')
plt.imshow(log_transformed, cmap='gray')

plt.show()
