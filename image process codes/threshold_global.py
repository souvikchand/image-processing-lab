import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply global thresholding
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(global_thresh, cmap='gray')
ax[1].set_title('Global Thresholding')
ax[1].axis('off')

plt.show()
