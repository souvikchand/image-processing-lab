import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply mean adaptive thresholding
mean_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

# Apply Gaussian adaptive thresholding
gaussian_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

# Display the original and thresholded images
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(mean_thresh, cmap='gray')
ax[1].set_title('Mean Adaptive Thresholding')
ax[1].axis('off')

ax[2].imshow(gaussian_thresh, cmap='gray')
ax[2].set_title('Gaussian Adaptive Thresholding')
ax[2].axis('off')

plt.show()
