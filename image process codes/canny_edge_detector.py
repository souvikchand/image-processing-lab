import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an example grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Canny Edge Detector
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Display the original image and the edge-detected image
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(edges, cmap='gray')
ax[1].set_title('Canny Edge Detection')
ax[1].axis('off')

plt.show()
