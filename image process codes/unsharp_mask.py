import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# Blur the image
blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
# Create the mask
mask = image - blurred
# Sharpen the image by adding the mask scaled by a factor
sharp_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Mask (High-pass)')
plt.imshow(mask, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Unsharp Masking')
plt.imshow(sharp_image, cmap='gray')
plt.show()