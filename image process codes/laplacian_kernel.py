import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# Apply Laplacian filter
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
laplacian_filtered = cv2.filter2D(image, -1, laplacian_kernel)
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Laplacian Sharpened')
plt.imshow(laplacian_filtered, cmap='gray')
plt.show()