import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('input_image.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the average brightness
avg_brightness = np.mean(gray_image)

# Define a target brightness level
target_brightness = 128

# Calculate the adjustment factor
adjustment_factor = target_brightness / avg_brightness

# Adjust the brightness
adjusted_image = cv2.convertScaleAbs(image, alpha=adjustment_factor, beta=0)

# Display the original and adjusted images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Brightness Adjusted Image')
ax[1].axis('off')

plt.show()
