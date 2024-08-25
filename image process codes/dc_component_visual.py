import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute Fourier transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Fourier Transform (Magnitude Spectrum)')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.colorbar()
plt.show()
