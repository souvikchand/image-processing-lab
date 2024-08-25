import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an example grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Harris Corner Detector
corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)

# Dilate corner image to enhance corner points
corners = cv2.dilate(corners, None)

# Threshold for an optimal value, marking the corners in red
image[corners > 0.01 * corners.max()] = 255

# Display the original image with detected corners
plt.imshow(image, cmap='gray')
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()
