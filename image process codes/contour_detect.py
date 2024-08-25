import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load an example grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a binary threshold to the image
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
contour_image = np.copy(image)
contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the original image and the image with detected contours
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(contour_image)
ax[1].set_title('Contour Detection')
ax[1].axis('off')

plt.show()
