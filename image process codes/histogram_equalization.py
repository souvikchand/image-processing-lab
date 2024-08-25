import cv2
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display the original and equalized images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(equalized_image, cmap='gray')
ax[1].set_title('Histogram Equalized Image')
ax[1].axis('off')

plt.show()
