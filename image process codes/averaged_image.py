import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply averaging filter
averaged = cv2.blur(image, (3, 3)) # 3x3 averaging kernel
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Averaged Image')
plt.imshow(averaged, cmap='gray')
plt.show()