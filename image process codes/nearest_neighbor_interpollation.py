import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an example grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Define the new size
new_size = (image.shape[1] * 2, image.shape[0] * 2)

# Resize the image using nearest-neighbor interpolation
resized_nearest = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

# Display the original and resized images
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(resized_nearest, cmap='gray')
ax[1].set_title('Resized Image (Nearest-Neighbor)')
ax[1].axis('off')

plt.show()
