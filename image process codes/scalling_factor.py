import cv2
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('input_image.jpg', cv2.IMREAD_COLOR)

# Define scaling factors
scale_x = 1.5  # Scale factor for width
scale_y = 1.5  # Scale factor for height

# Resize image using OpenCV
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display the original and scaled images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Scaled Image')
ax[1].axis('off')

plt.show()
