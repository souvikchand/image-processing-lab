import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Fourier Transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency component to center

# Apply high-pass filter in frequency domain
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
D = 30  # cutoff radius
H = np.ones((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
        H[i, j] = (1 - np.exp(-0.5 * (dist**2 / (D**2))))

# Apply filter
filtered_f_transform = f_transform_shifted * H

# Inverse Fourier Transform
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_f_transform))
filtered_image = np.abs(filtered_image)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('High-pass Filtered Image')
plt.imshow(filtered_image, cmap='gray')

plt.show()
