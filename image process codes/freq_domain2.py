import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Fourier Transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency component to center

# Compute magnitude spectrum (log scale for visualization)
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Compute phase spectrum
phase_spectrum = np.angle(f_transform_shifted)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Magnitude Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Phase Spectrum')
plt.imshow(phase_spectrum, cmap='gray')

plt.show()
