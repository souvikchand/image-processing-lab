import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
image_float = np.float32(image) / 255.0

# Apply logarithmic transformation
log_image = np.log1p(image_float)

# Apply Fourier Transform
f_transform = np.fft.fft2(log_image)

# Define high-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
D = 30  # cutoff radius
H = np.ones((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
        H[i, j] = (1 - np.exp(-0.5 * (dist**2 / (D**2))))

# Apply high-pass filter in frequency domain
filtered_f_transform = f_transform * H

# Apply inverse Fourier Transform
filtered_image = np.fft.ifft2(filtered_f_transform)

# Exponentiate to revert the logarithmic transformation
filtered_image = np.exp(np.real(filtered_image))

# Normalize to [0, 255]
filtered_image = np.uint8(filtered_image * 255)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Homomorphic Filtered Image')
plt.imshow(filtered_image, cmap='gray')

plt.show()
