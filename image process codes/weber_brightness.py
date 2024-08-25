import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define the Weber fraction for brightness perception
weber_fraction = 0.02

# Function to apply Weber's Law for brightness adjustment
def adjust_brightness(image, adjustment_factor):
    adjusted_image = image + (weber_fraction * adjustment_factor * image)
    adjusted_image = np.clip(adjusted_image, 0, 255)  # Ensure pixel values remain valid
    return adjusted_image.astype(np.uint8)

# Adjust the brightness of the image
adjustment_factor = 1  # Scale factor for brightness adjustment
brightened_image = adjust_brightness(image, adjustment_factor)

# Display the original and adjusted images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(brightened_image, cmap='gray')
ax[1].set_title('Brightened Image')
ax[1].axis('off')

plt.show()
